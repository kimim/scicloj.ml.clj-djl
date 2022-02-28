(ns scicloj.ml.clj-djl.fasttext-test
  (:require [scicloj.ml.clj-djl.fasttext :as fasttext]
            [clojure.test  :refer [deftest is] :as t]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset :as ds]))




(def primary
  (->
   (ds/->dataset "test/data/primary_small.nippy" {:key-fn keyword})))

(deftest fasttext-train

  (let [model
        (ml/train (-> primary
                      (tech.v3.dataset.modelling/set-inference-target :is.primary))
                  {:model-type :clj-djl/fasttext})


        prob-distribution
        (ml/predict (ds/head primary) (assoc model
                                             :top-k 3))]
    (is (= ["yes" "yes" "yes" "yes" "yes"] (prob-distribution :is.primary)))))
