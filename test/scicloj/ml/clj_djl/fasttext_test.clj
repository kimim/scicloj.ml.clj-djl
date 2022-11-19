(ns scicloj.ml.clj-djl.fasttext-test
  (:require
   [clojure.java.io :as io]
   [clojure.test :as t :refer [deftest is]]
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling]))

(defn delete-files-recursively
  [f1 & [silently]]
  (when (.isDirectory (io/file f1))
    (doseq [f2 (.listFiles (io/file f1))]
      (delete-files-recursively f2 silently)))
  (io/delete-file f1 silently))

(def primary
  (->
   (ds/->dataset "test/data/primary_small.nippy" {:key-fn keyword})))

(deftest fasttext-train

  (let [model
        (ml/train (-> primary
                      (tech.v3.dataset.modelling/set-inference-target :is.primary))
                  {:model-type :clj-djl/fasttext})
        ;; :ft-training-config {:epoch 1}



        prob-distribution
        (ml/predict (ds/head primary) (assoc model
                                             :top-k 3))]
    ;; (println :model-dir (io/file (-> model :model-data :model-dir)))
    (delete-files-recursively (io/file (-> model :model-data :model-dir)))
    (is (= ["yes" "yes" "yes" "yes" "yes"] (prob-distribution :is.primary)))))
