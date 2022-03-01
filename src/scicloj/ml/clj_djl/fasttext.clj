(ns scicloj.ml.clj-djl.fasttext
  (:require [clojure.java.io :as io]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.categorical :as ds-cat]
            [scicloj.metamorph.ml :as ml])
  (:import [ai.djl.fasttext FtModel FtTrainingConfig]
           [ai.djl.basicdataset.nlp CookingStackExchange]
           [ai.djl.basicdataset RawDataset]
           [java.nio.file.attribute FileAttribute]))

(defn make-dataset [path]
  (reify RawDataset
    (getData [this] path)))
            

(defn ->fast-text-file! [ft-ds out-file]
  (->> ft-ds
       :fast-text
       (clojure.string/join "\n")
       (spit (.toFile  out-file))))

(defn ->fast-text-ds [ds label-col text-col]

  (-> ds
      (tc/add-column :fast-text (fn [ds ] (map #(str  "__label__" %2 " " %1)

                                              (get ds text-col)
                                              (get ds label-col))))
      (tc/select-columns :fast-text)))


(defn train-ft [ds label-col text-col]
  (let [
        model-name "my-model"
        model (FtModel. "my-model")
        temp-dir  (java.nio.file.Files/createTempDirectory "fasttext"
                                                           (into-array FileAttribute []))
        fasttext-file (java.nio.file.Files/createTempFile "fasttext" ".txt"
                                                          (into-array FileAttribute []))
        training-config

        (..  (FtTrainingConfig/builder)
             (setModelName model-name)
             (setOutputDir temp-dir)
             build)
        model-file (str (.. temp-dir  toFile getPath)
                    "/"
                    model-name  ".bin")]
    (-> ds
        (->fast-text-ds label-col text-col)
        (->fast-text-file! fasttext-file))
    (.. model
        (fit training-config (make-dataset fasttext-file))) ;; (.. (CookingStackExchange/builder) build)


    {
     :model-file model-file}))


(defn ->maps [classification]
  (->>
   (.items classification)
   (map #(hash-map :class-name (.getClassName %)
                   :probability (.getProbability %)))))

(defn classify [model text top-k]
  (let [raw-classification (->maps (.classify model text top-k))]
    (if-not (empty? raw-classification)
      raw-classification
      [{:class-name nil
        :probability 0}])))

(defn predict-ft [feature-ds model top-k]
  (let [
        texts (->  (tc/columns feature-ds :as-seq) first seq)]

    (map
     #(classify model (str %) top-k)
     texts)))


(defn train
  [feature-ds label-ds options]

  (let [label-columns (tc/column-names label-ds)
        feature-columns (tc/column-names feature-ds)]

    (when (not  (= (count feature-columns)  1))
      (throw (ex-info "Dataset should have exactly one feature column." {:columns feature-columns})))
    (when (not  (= (count label-columns) 1))
      (throw (ex-info "Dataset should have exactly one target column." {:columns label-columns})))
    (train-ft (tc/append feature-ds label-ds)
              (first label-columns)
              (first feature-columns))))


(defn predict
  [feature-ds thawed-model {:keys [target-columns
                                   target-categorical-maps
                                   top-k
                                   options]}]
  (assert  (= 1 (tc/column-count feature-ds)) "Dataset should have exactly one column.")

  (let [top-k (or top-k 2)

        _ (assert (> top-k 1) "top-k need to be at least 2")

        ft-prediction
        (->>
         (predict-ft feature-ds thawed-model top-k)
         (map
          #(map (fn [m] (assoc m :id %1)) %2)
          (range)))


        predictions-ds
        (->
         ft-prediction
         flatten
         ds/->dataset
         (tc/pivot->wider [:class-name] [:probability] {:drop-missing? false})
         (tc/order-by :id)
         (ds/drop-columns [:id]))


        predictions-with-label
        (->
         (tech.v3.dataset.modelling/probability-distributions->label-column
          predictions-ds
          (first target-columns))
         (ds-cat/reverse-map-categorical-xforms))]

    predictions-with-label))

(defn load-ft-model [path]
  (prn :load-model path)
  (let [model-instance (FtModel. "my-model")]

    (.load model-instance (.toPath (io/file path)))
    model-instance))

(ml/define-model! :clj-djl/fasttext train predict
   {:thaw-fn (fn [model]
               (load-ft-model (:model-file model)))})
