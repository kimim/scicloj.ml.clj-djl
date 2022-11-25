(ns scicloj.ml.clj-djl.fasttext
  (:require
   [camel-snake-kebab.core :as csk]
   [clojure.java.io :as io]
   [clojure.reflect]
   [clojure.string]
   [scicloj.metamorph.ml :as ml]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.modelling])
  (:import
   (ai.djl.basicdataset RawDataset)
   (ai.djl.fasttext FtModel FtTrainingConfig TrainFastText)
   (java.nio.file.attribute FileAttribute)))

(defn opts-docu []
  (->>
   (FtTrainingConfig/builder)
   (clojure.reflect/reflect)
   :members
   (filter #(clojure.string/starts-with? (:name %) "opt"))
   (map #(hash-map :name (csk/->kebab-case-keyword (clojure.string/replace (:name  %) "opt" ""))
                   :type (first  (:parameter-types %))))))

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


(defn do-opts [config m]
  (doseq [[k v] m]
    (let [method-name (str "opt" (csk/->PascalCaseString k))]
      (clojure.lang.Reflector/invokeInstanceMethod
        config
        method-name
        (into-array Object [v]))))
  config)


(defn train-ft [ds label-col text-col ft-training-config]
  (let [

        model-name "my-model"
        temp-dir  (java.nio.file.Files/createTempDirectory "fasttext"
                                                           (into-array FileAttribute []))
        fasttext-file (java.nio.file.Files/createTempFile "fasttext" ".txt"
                                                          (into-array FileAttribute []))

        training-config
        (->
         (..  (FtTrainingConfig/builder)
              (setModelName model-name)
              (setOutputDir temp-dir))
         (do-opts ft-training-config)
         .build)

        model-file (str (.. temp-dir  toFile getPath)
                        "/"
                        model-name  ".bin")
        _ (-> ds
              (->fast-text-ds label-col text-col)
              (->fast-text-file! fasttext-file))


        _ (.. temp-dir toFile deleteOnExit)
        ft-text-classification (TrainFastText/textClassification training-config (make-dataset fasttext-file))]

    (java.nio.file.Files/delete fasttext-file)
    (.close ft-text-classification)
    {
     :classes (->>
               (get ds label-col)
               distinct
               (into #{}))
     :model-file model-file
     :model-dir (.getAbsolutePath (.toFile temp-dir))}))



(defn ->maps [classification]
  (->>
   (.items classification)
   (map #(hash-map :class-name (.getClassName %)
                   :probability (.getProbability %)))))


(defn classify [model text top-k classes]
  (let [raw-classification (->maps (.. model
                                       getBlock
                                       (classify text top-k)))]
    (if-not (empty? raw-classification)
      raw-classification
      (map
       #(hash-map
         :class-name (str %)
         :probability (/ 1 (count classes)))
       classes))))

(defn predict-ft [feature-ds model top-k classes]
  (let [
        texts (->  (tc/columns feature-ds :as-seq) first seq)]

    (map
     #(classify model (str %) top-k classes)
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
              (first feature-columns)
              (get options :ft-training-config {}))))


(defn predict
  [feature-ds thawed-model {:keys [target-columns
                                   target-categorical-maps
                                   top-k
                                   options] :as model}]
  (assert  (= 1 (tc/column-count feature-ds)) "Dataset should have exactly one column.")



  (let [target-colname (first target-columns)
        top-k (or top-k (-> target-categorical-maps (get target-colname) :lookup-table count))

        _ (assert (> top-k 1) "top-k need to be at least 2")

        _ (assert (:model thawed-model) "No model given in 'thawed-model'")
        _ (assert (:classes thawed-model) "No classes in thawed mode")
        ft-prediction
        (->>
         (predict-ft feature-ds (:model thawed-model) top-k (:classes thawed-model))
         (map
          #(map (fn [m] (assoc m :id %1)) %2)
          (range)))


        predictions-ds
        (->
         ft-prediction
         flatten
         ds/->dataset
         (tc/add-or-replace-column :class-name (fn [ds] (map #(Integer/parseInt %) (:class-name ds))))
         (tc/pivot->wider [:class-name] [:probability] {:drop-missing? false})
         (tc/order-by :id)
         (ds/drop-columns [:id]))

        predictions-with-label-index
        (->
         (tech.v3.dataset.modelling/probability-distributions->label-column
          predictions-ds
          target-colname))
         

        index->label-fn (zipmap (range)
                                (map #(Integer/parseInt %)  (tc/column-names predictions-ds)))
        predictions-with-label
        (-> predictions-with-label-index
            (ds/update-column target-colname #(map int %))
            (ds/add-or-update-column target-colname (map index->label-fn (get predictions-with-label-index target-colname)))
            (ds/update-column target-colname
                              #(vary-meta % assoc :categorical-map (get target-categorical-maps target-colname))))]


    (.close (:model thawed-model))
    (-> predictions-with-label
        (ds/update-column target-colname
                          #(vary-meta % assoc :column-type :prediction)))))

(defn load-ft-model [path]
  (let [model-instance (FtModel. "my-model")]

    (.load model-instance (.toPath (io/file path)))
    model-instance))

(ml/define-model! :clj-djl/fasttext train predict
  {:documentation {:javadoc    "https://javadoc.io/doc/ai.djl.fasttext/fasttext-engine/latest/index.html"
                   :user-guide "https://djl.ai/extensions/fasttext/"}
   :options (opts-docu)
   :thaw-fn (fn [model]
              (assert (:model-file model) ":model-file not given in `model`")
              (assoc model :model
                     (load-ft-model (:model-file model))))})
