(ns clj-djl.mmml-test
  (:require [clj-djl.mmml]
            [scicloj.ml.dataset :as ds]
            [scicloj.ml.metamorph :as mm]
            [scicloj.ml.core :as ml]
            [scicloj.metamorph.ml.categorical]
            [clojure.test :refer [deftest is]]
            [clj-djl.nn :as nn]
            [clj-djl.training :as t]
            [clj-djl.training.loss :as loss]
            [clj-djl.training.optimizer :as optimizer]
            [clj-djl.training.tracker :as tracker]
            [clj-djl.training.listener :as listener]
            [clj-djl.ndarray :as nd]
            [tech.v3.datatype.functional :as dfn]
            [clj-djl.nn.parameter :as param]))

(tablecloth.pipeline/update-columns)
(defn count-small [seq]
  (count
   (filter

    #(and (> 5 %)
          (<= -5 %))
    seq)))


(def train-ds
  (tech.v3.dataset/->dataset
   "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv"))


(def test-ds
  (->
   (tech.v3.dataset/->dataset
    "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv")
   (tech.v3.dataset/add-column (tech.v3.dataset/new-column  "SalePrice" [0]))))


(defn numeric-features [ds]
  (def ds ds)
  (tech.v3.dataset.column-filters/intersection (tech.v3.dataset.column-filters/numeric ds)
                                               (tech.v3.dataset.column-filters/feature ds)))

(defn update-columns
  "Update a sequence of columns selected by column name seq or column selector function."
  [dataframe col-name-seq-or-fn update-fn]
  (tech.v3.dataset/update-columns dataframe
                                  (if (fn? col-name-seq-or-fn)
                                    (tech.v3.dataset/column-names (col-name-seq-or-fn dataframe))
                                    col-name-seq-or-fn)
                                  update-fn))



(def  learning-rate 0.01)
(defn net [] (nn/sequential {:blocks (nn/linear {:units 1})
                             :initializer (nn/normal-initializer)
                             :parameter param/weight}))
(defn cfg [] (t/training-config {:loss (loss/l2-loss)
                                 :optimizer (optimizer/sgd
                                             {:tracker (tracker/fixed learning-rate)})
                                 :evaluator (t/accuracy)
                                 :listeners (listener/logging)}))


(deftest train-predict-1

  (let [preprocess
        (fn [ds ds-indices]
         (-> ds
             (tech.v3.dataset/drop-columns ["Id"])
             (tech.v3.dataset.modelling/set-inference-target "SalePrice")

             (tech.v3.dataset/replace-missing tech.v3.dataset.column-filters/numeric :value 0)
             (tech.v3.dataset/replace-missing tech.v3.dataset.column-filters/categorical :value "None")
             (tech.v3.dataset.modelling/set-inference-target "SalePrice")
             (update-columns numeric-features
                             #(dfn// (dfn/- % (dfn/mean %))
                                     (dfn/standard-deviation %)))
             (update-columns ["SalePrice"]
                   #(dfn// % (dfn/mean %)))
             (tech.v3.dataset.modelling/set-inference-target "SalePrice")
             (tech.v3.dataset/categorical->one-hot tech.v3.dataset.column-filters/categorical)
             (tech.v3.dataset/select-rows ds-indices)))

        trained-model
        (-> (preprocess (tech.v3.dataset/concat train-ds test-ds)
                        (range (tech.v3.dataset/row-count train-ds)))
            ((fn [ds]
               (def ds-1 ds)
               ds))
            (scicloj.metamorph.ml/train {:model-type :clj-djl/djl
                                         :batchsize 64
                                         :model-spec {:name "mlp" :block-fn net}
                                         :model-cfg (cfg)
                                         :initial-shape (nd/shape 1 310)
                                         :nepoch 1}))
        prediction
        (-> (preprocess (tech.v3.dataset/concat  test-ds train-ds)
                        (range (tech.v3.dataset/row-count test-ds)))
            (scicloj.metamorph.ml/predict trained-model))]

    (is (= 1458
           (count-small
            (get prediction "SalePrice"))))))


(defn update-columns-by-meta [column-selector update-fn]
  (fn [ctx]
    (let [col-names (ds/column-names (ctx :metamorph/data) column-selector :all)]
      (update ctx :metamorph/data #(ds/update-columns % col-names update-fn)))))

(defn numeric-feature? [meta]
  (and
   (tablecloth.api.utils/type? :numerical (meta :datatype))
   (not (meta :inference-target?))))

(deftest train-predict-2
  (let [ preprocss-pipe-fn
        (ml/pipeline
         (mm/drop-columns ["Id"])
         (mm/replace-missing :type/numerical :value 0)
         (mm/replace-missing :!type/numerical :value "None")
         (mm/set-inference-target "SalePrice"
         ;; (mm/update-columns numeric-feature? :all (fn [col] (dfn// (dfn/- col (dfn/mean col)
         ;;                                                            (dfn/standard-deviation col)))))
          (update-columns-by-meta numeric-feature?
                                  (fn [col] (dfn// (dfn/- col (dfn/mean col)
                                                         (dfn/standard-deviation col))))))
         (mm/set-inference-target "SalePrice")
         (mm/update-column "SalePrice" #(dfn// % (dfn/mean %)))
         (mm/set-inference-target "SalePrice"))

        preprocessed-full-ds (:metamorph/data (ml/fit-pipe (ds/concat train-ds test-ds) preprocss-pipe-fn))



        final-pipe-fn
        (ml/pipeline
         preprocss-pipe-fn
         (scicloj.metamorph.ml.categorical/transform-one-hot
          :!type/numerical :full)
         (mm/model {:model-type :clj-djl/djl
                    :batchsize 64
                    :model-spec {:name "mlp" :block-fn net}
                    :model-cfg (cfg)
                    :initial-shape (nd/shape 1 310)
                    :nepoch 10}))

        fit-ctx
        (final-pipe-fn {:metamorph/data train-ds
                        :metamorph/mode :fit
                        :metamorph.ml/full-ds preprocessed-full-ds})

        transform-result
        (final-pipe-fn (merge fit-ctx
                              {:metamorph/data test-ds
                               :metamorph/mode :transform
                               :metamorph.ml/full-ds preprocessed-full-ds}))]

    (is (= 1459
           (count-small
            (-> transform-result :metamorph/data (get "SalePrice")))))))
