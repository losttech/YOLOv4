namespace tensorflow.keras.applications {
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using LostTech.Gradient;
    using LostTech.Gradient.BuiltIns;

    using numpy;

    using tensorflow.data;
    using tensorflow.keras.callbacks;
    using tensorflow.keras.models;
    using tensorflow.keras.optimizers;
    using tensorflow.summary;

    public static class YOLO {
        public static void Train(Model model, IOptimizer optimizer, ObjectDetectionDataset dataset,
                                 ObjectDetectionDataset? testSet = null,
                                 IEnumerable<ICallback>? callbacks = null,
                                 int batchSize = 2,
                                 int warmupEpochs = 2, int firstStageEpochs = 20,
                                 int secondStageEpochs = 30,
                                 float initialLearningRate = 1e-3f,
                                 float finalLearningRate = 1e-6f,
                                 ISummaryWriter? summaryWriter = null) {
            long globalSteps = 1;

            int warmupSteps = warmupEpochs * dataset.Count;
            long totalSteps = (long)(firstStageEpochs + secondStageEpochs) * dataset.Count;

            foreach (var callback in callbacks ?? Array.Empty<ICallback>()) {
                callback.DynamicInvoke<object>("set_model", model);
                callback.DynamicInvoke<object>("set_params", new Dictionary<string, object> {
                    ["metrics"] = new[] { "loss", "testLoss" }
                    .SelectMany(prefix => new[]{
                        prefix + nameof(Loss.GIUO),
                        prefix + nameof(Loss.Conf),
                        prefix + nameof(Loss.Prob),
                    }).ToArray(),
                });
            }

            Loss TrainStep(ObjectDetectionDataset.EntryBatch batch) {
                var tape = new GradientTape();
                Loss losses;
                Tensor totalLoss;
                using (tape.StartUsing()) {
                    losses = ComputeLosses(model, batch, dataset.ClassNames.Length, dataset.Strides);
                    totalLoss = losses.GIUO + losses.Conf + losses.Prob;

                    if (!tf.executing_eagerly() || !tf.logical_or(tf.is_inf(totalLoss), tf.is_nan(totalLoss)).numpy().any()) {
                        PythonList<Tensor> gradients = tape.gradient(totalLoss, model.trainable_variables);
                        optimizer.apply_gradients(gradients.Zip(
                            (PythonList<Variable>)model.trainable_variables, (g, v) => (g, v)));
                    } else {
                        System.Diagnostics.Debug.WriteLine("NaN/inf loss ignored");
                    }
                }

                globalSteps++;

                double learningRate = globalSteps < warmupSteps
                    ? globalSteps / (float)warmupSteps * initialLearningRate
                    : finalLearningRate + 0.5f * (initialLearningRate - finalLearningRate) * (
                        1 + Math.Cos((globalSteps - warmupSteps) / (totalSteps - warmupSteps) * Math.PI)
                    );
                var optimizerLearningRate = optimizer.DynamicGet<Variable>("lr");
                optimizerLearningRate.assign_dyn(tf.constant(learningRate));
                if (summaryWriter != null) {
                    var activeWriter = summaryWriter.as_default();
                    activeWriter.__enter__();
                    tf.summary.experimental.set_step(tf.constant(globalSteps));
                    tf.summary.scalar("lr", learningRate);
                    tf.summary.scalar("loss/total_loss", totalLoss);
                    tf.summary.scalar("loss/giou_loss", losses.GIUO);
                    tf.summary.scalar("loss/conf_loss", losses.Conf);
                    tf.summary.scalar("loss/prob_loss", losses.Prob);
                    activeWriter.__exit__(null, null, null);

                    summaryWriter.flush();
                }

                return losses;
            }

            Loss TestStep(ObjectDetectionDataset.EntryBatch batch) {
                var output = model.__call___dyn(batch.Images, new { training = true }.AsKwArgs());
                return ComputeLosses(model, batch, dataset.ClassNames.Length, dataset.Strides);
            }

            bool isFreeze = false;
            string[] freezeLayers = { "conv2d_93", "conv2d_101", "conv2d_109" };
            void SetFreeze(bool freeze) {
                foreach(string name in freezeLayers) {
                    var layer = model.get_layer(name);
                    Utils.SetTrainableRecursive(layer, !freeze);
                }
            }
            foreach (int epoch in Enumerable.Range(0, firstStageEpochs + secondStageEpochs)) {
                if (epoch < firstStageEpochs) {
                    if (!isFreeze) {
                        isFreeze = true;

                        SetFreeze(true);
                    }
                } else {
                    if (isFreeze) {
                        isFreeze = false;
                        SetFreeze(false);
                    }
                }

                foreach (var callback in callbacks ?? Array.Empty<ICallback>())
                    callback.on_epoch_begin(epoch);

                var trainLoss = Loss.Zero;
                foreach (var batch in dataset.Batch(batchSize: batchSize,
                                                    onloadAugmentation: ObjectDetectionDataset.RandomlyApplyAugmentations)
                                      .BufferedEnumerate(bufferSize: 6)) {
                    trainLoss += TrainStep(batch);
                }

                var testLoss = Loss.Zero;
                if (testSet != null) {
                    foreach (var batch in testSet.Batch(batchSize: batchSize, onloadAugmentation: null))
                        testLoss += TestStep(batch);
                }

                foreach (var callback in callbacks ?? Array.Empty<ICallback>()) {
                    var logs = new Dictionary<string, object?>();
                    (trainLoss / dataset.Count).Write(logs, "loss");
                    if (testSet != null)
                        (testLoss / testSet.Count).Write(logs, "testLoss");
                    callback.on_epoch_end(epoch, logs: logs);
                }
            }
        }

        public static Loss ComputeLosses(Model model,
                                         ObjectDetectionDataset.EntryBatch batch,
                                         int classCount, ReadOnlySpan<int> strides) {
            if (model is null) throw new ArgumentNullException(nameof(model));
            if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));

            var output = model.__call___dyn(batch.Images, new { training = true }.AsKwArgs());
            var loss = Loss.Zero;
            for (int scaleIndex = 0; scaleIndex < YOLOv4.XYScale.Length; scaleIndex++) {
                Tensor conv = output[scaleIndex * 2];
                Tensor pred = output[scaleIndex * 2 + 1];

                loss += ComputeLoss(pred, conv,
                                    targetLabels: batch.BBoxLabels[scaleIndex],
                                    targetBBoxes: batch.BBoxes[scaleIndex],
                                    strideSize: strides[scaleIndex],
                                    classCount: classCount,
                                    intersectionOverUnionLossThreshold: DefaultIntersectionOverUnionLossThreshold);
            }

            return loss;
        }

        public static Model CreateV4(int inputSize, int classCount, ReadOnlySpan<int> strides) {
            if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
            if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));

            Tensor input = tf.keras.Input(new TensorShape(inputSize, inputSize, 3));
            var featureMaps = YOLOv4.Apply(input, classCount: classCount);

            var bboxTensors = new PythonList<Tensor>();
            foreach (var (scaleIndex, featureMap) in Tools.Enumerate(featureMaps.SSBox, featureMaps.MBBox, featureMaps.LBBox)) {
                var bbox = DecodeTrain(featureMap, classCount: classCount,
                    anchors: YOLOv4.Anchors, strides: strides,
                    scaleIndex: scaleIndex, xyScale: YOLOv4.XYScale);
                bboxTensors.Add(featureMap);
                bboxTensors.Add(bbox);
            }
            return new Model(new { inputs = input, outputs = bboxTensors }.AsKwArgs());
        }

        public struct Loss {
            public Tensor GIUO { get; set; }
            public Tensor Conf { get; set; }
            public Tensor Prob { get; set; }

            public static Loss operator +(Loss a, Loss b) => new Loss {
                GIUO = a.GIUO + b.GIUO,
                Conf = a.Conf + b.Conf,
                Prob = a.Prob + b.Prob,
            };
            public static Loss operator /(Loss a, float divisor) => new Loss {
                GIUO = a.GIUO / divisor,
                Conf = a.Conf / divisor,
                Prob = a.Prob / divisor,
            };

            public static Loss Zero => new Loss {
                GIUO = tf.constant(0f),
                Conf = tf.constant(0f),
                Prob = tf.constant(0f),
            };

            public void Write(IDictionary<string, object?> metrics, string prefix = "") {
                if (metrics is null) throw new ArgumentNullException(nameof(metrics));
                if (prefix is null) throw new ArgumentNullException(nameof(prefix));

                metrics[prefix + nameof(this.GIUO)] = this.GIUO;
                metrics[prefix + nameof(this.Conf)] = this.Conf;
                metrics[prefix + nameof(this.Prob)] = this.Prob;
            }
        }

        static Loss ComputeLoss(Tensor pred, Tensor conv,
                                ndarray<float> targetLabels, ndarray<float> targetBBoxes,
                                int strideSize, int classCount,
                                float intersectionOverUnionLossThreshold) {
            int batchSize = conv.shape[0];
            int outputSize = conv.shape[1];
            float inputSize = strideSize * outputSize;

            conv = tf.reshape(conv, new[] { batchSize, outputSize, outputSize, 3, 5 + classCount });

            var convRawConf = conv[.., .., .., .., 4..5];
            var convRawProb = conv[.., .., .., .., 5..];

            var predXYWH = pred[.., .., .., .., 0..4];
            var predConf = pred[.., .., .., .., 4..5];

            var labelXYWH = tf.constant(targetLabels[.., .., .., .., 0..4]);
            var respondBBox = tf.constant(targetLabels[.., .., .., .., 4..5]);
            var labelProb = tf.constant(targetLabels[.., .., .., .., 5..]);

            var generalizedIntersectionOverUnion = tf.expand_dims(
                BBoxGeneralizedIntersectionOverUnion(predXYWH, labelXYWH),
                axis: - 1);

            Tensor bboxLossScale = 2f - 1f * labelXYWH[.., .., .., .., 2..3] * labelXYWH[.., .., .., .., 3..4] / (inputSize * inputSize);
            Tensor generalizedIntersectionOverUnionLoss =
                respondBBox * bboxLossScale * (1 - generalizedIntersectionOverUnion);

            var intersectionOverUnion = BBoxIOU(
                boxes1: predXYWH[.., .., .., .., tf.newaxis, ..],
                boxes2: tf.constant(targetBBoxes[.., np.newaxis, np.newaxis, np.newaxis, .., ..]));

            var maxIntersectionOverUnion = tf.expand_dims(
                tf.reduce_max(intersectionOverUnion, axis: new[] { -1 }),
                axis: -1);

            var respondBackground = (1f - respondBBox) * tf.cast(maxIntersectionOverUnion < intersectionOverUnionLossThreshold, tf.float32);

            var confFocal = tf.pow(respondBBox - predConf, 2);
            Tensor confLoss = confFocal * (
                respondBBox * tf.nn.sigmoid_cross_entropy_with_logits(labels: respondBBox, logits: convRawConf)
                +
                respondBackground * tf.nn.sigmoid_cross_entropy_with_logits(labels: respondBBox, logits: convRawConf));

            Tensor probLoss = respondBBox * tf.nn.sigmoid_cross_entropy_with_logits(labels: labelProb, logits: convRawProb);

            generalizedIntersectionOverUnionLoss = tf.reduce_mean(tf.reduce_sum(generalizedIntersectionOverUnionLoss, axis: new[] { 1, 2, 3, 4 }));
            confLoss = tf.reduce_mean(tf.reduce_sum(confLoss, axis: new[] { 1, 2, 3, 4 }));
            probLoss = tf.reduce_mean(tf.reduce_sum(probLoss, axis: new[] { 1, 2, 3, 4 }));

            return new Loss {
                GIUO = generalizedIntersectionOverUnionLoss,
                Conf = confLoss,
                Prob = probLoss,
            };
        }

        static Tensor BBoxIOU(Tensor boxes1, Tensor boxes2) {
            var area1 = boxes1[tf.rest_of_the_axes, 2] * boxes1[tf.rest_of_the_axes, 3];
            var area2 = boxes1[tf.rest_of_the_axes, 2] * boxes1[tf.rest_of_the_axes, 3];

            boxes1 = tf.concat(new[] {
                boxes1[tf.rest_of_the_axes, ..2] - boxes1[tf.rest_of_the_axes, 2..] * 0.5f,
                boxes1[tf.rest_of_the_axes, ..2] + boxes1[tf.rest_of_the_axes, 2..] * 0.5f,
            }, axis: -1);
            boxes2 = tf.concat(new[] {
                boxes2[tf.rest_of_the_axes, ..2] - boxes2[tf.rest_of_the_axes, 2..]*0.5f,
                boxes2[tf.rest_of_the_axes, ..2] + boxes2[tf.rest_of_the_axes, 2..]*0.5f,
            }, axis: -1);

            Tensor leftUp = tf.maximum(boxes1[tf.rest_of_the_axes, ..2], boxes2[tf.rest_of_the_axes, ..2]);
            Tensor rightDown = tf.minimum(boxes1[tf.rest_of_the_axes, 2..], boxes2[tf.rest_of_the_axes, 2..]);

            var intersection = tf.maximum(rightDown - leftUp, 0.0f);
            var intersectionArea = intersection[tf.rest_of_the_axes, 0] * intersection[tf.rest_of_the_axes, 1];
            var unionArea = area1 + area2 - intersectionArea;

            return intersectionArea / unionArea;
        }
        static Tensor BBoxGeneralizedIntersectionOverUnion(Tensor boxes1, Tensor boxes2) {
            boxes1 = tf.concat(new[]{
                boxes1[tf.rest_of_the_axes, ..2] - boxes1[tf.rest_of_the_axes, 2..] * 0.5f,
                boxes1[tf.rest_of_the_axes, ..2] + boxes1[tf.rest_of_the_axes, ..2] * 0.5f,
                }, axis: -1);
            boxes2 = tf.concat(new[]{
                boxes2[tf.rest_of_the_axes, ..2] - boxes2[tf.rest_of_the_axes, 2..] * 0.5f,
                boxes2[tf.rest_of_the_axes, ..2] + boxes2[tf.rest_of_the_axes, ..2] * 0.5f,
                }, axis: -1);

            boxes1 = tf.concat(new[] {
                tf.minimum(boxes1[tf.rest_of_the_axes, ..2], boxes1[tf.rest_of_the_axes, 2..]),
                tf.maximum(boxes1[tf.rest_of_the_axes, ..2], boxes1[tf.rest_of_the_axes, 2..]),
            }, axis: -1);
            boxes2 = tf.concat(new[] {
                tf.minimum(boxes2[tf.rest_of_the_axes, ..2], boxes2[tf.rest_of_the_axes, 2..]),
                tf.maximum(boxes2[tf.rest_of_the_axes, ..2], boxes2[tf.rest_of_the_axes, 2..]),
            }, axis: -1);

            var boxes1Area = BoxesArea(boxes1);
            var boxex2Area = BoxesArea(boxes2);

            Tensor leftUp = tf.maximum(boxes1[tf.rest_of_the_axes, ..2], boxes2[tf.rest_of_the_axes, ..2]);
            Tensor rigthDown = tf.minimum(boxes1[tf.rest_of_the_axes, 2..], boxes2[tf.rest_of_the_axes, 2..]);

            Tensor intersection = tf.maximum(rigthDown - leftUp, 0);
            Tensor intersectionArea = intersection[tf.rest_of_the_axes, 0] * intersection[tf.rest_of_the_axes, 1];
            Tensor unionArea = boxes1Area + boxex2Area - intersectionArea;
            Tensor intersectionOverUnion = intersectionArea / unionArea;

            Tensor encloseLeftUp = tf.minimum(boxes1[tf.rest_of_the_axes, ..2], boxes2[tf.rest_of_the_axes, ..2]);
            Tensor encloseRightDown = tf.maximum(boxes1[tf.rest_of_the_axes, 2..], boxes2[tf.rest_of_the_axes, 2..]);
            Tensor enclose = tf.maximum(encloseRightDown - encloseLeftUp, 0);
            Tensor encloseArea = enclose[tf.rest_of_the_axes, 0] * enclose[tf.rest_of_the_axes, 1];

            var generalized = intersectionOverUnion - 1f * (encloseArea - unionArea) / encloseArea;
            return generalized;
        }

        static Tensor BoxesArea(Tensor boxes)
            => (boxes[tf.rest_of_the_axes, 2] - boxes[tf.rest_of_the_axes, 0])
                * (boxes[tf.rest_of_the_axes, 3] - boxes[tf.rest_of_the_axes, 1]);

        static Tensor DecodeTrain(Tensor convOut, int classCount,
                                  ReadOnlySpan<int> strides, ReadOnlySpan<int> anchors,
                                  int scaleIndex, ReadOnlySpan<float> xyScale) {
            var varScope = new variable_scope("scale" + scaleIndex.ToString(System.Globalization.CultureInfo.InvariantCulture));
            using var _ = varScope.StartUsing();
            Tensor batchSize = tf.shape(convOut)[0];
            Tensor outputSize = tf.shape(convOut)[1];

            convOut = tf.reshape_dyn(convOut, new object[] { batchSize, outputSize, outputSize, 3, 5 + classCount });
            Tensor[] raws = tf.split(convOut, new[] { 2, 2, 1, classCount }, axis: -1);
            var (convRawDxDy, convRawDwDh, convRawConf, convRawProb) = raws;

            Tensor x = tf.tile_dyn(tf.expand_dims(tf.range(outputSize, dtype: tf.int32), axis: 0),
                                   new object[] { outputSize, 1 });
            Tensor y = tf.tile_dyn(tf.expand_dims(tf.range(outputSize, dtype: tf.int32), axis: 1),
                                   new object[] { 1, outputSize });
            Tensor xyGrid = tf.expand_dims(tf.stack(new[] { x, y }, axis: -1), axis: 2); // [gx, gy, 1, 2]

            xyGrid = tf.tile_dyn(tf.expand_dims(xyGrid, axis: 0),
                                 new object[] { batchSize, 1, 1, 3, 1 });
            xyGrid = tf.cast(xyGrid, tf.float32);

            var predictedXY = ((tf.sigmoid(convRawDxDy) * xyScale[scaleIndex]) - 0.5 * (xyScale[scaleIndex] - 1) + xyGrid) * strides[scaleIndex];
            var predictedWH = (tf.exp(convRawDwDh) * anchors[scaleIndex]);
            var predictedXYWH = tf.concat(new[] { predictedXY, predictedWH }, axis: -1);

            var predictedConf = tf.sigmoid(convRawConf);
            var predictedProb = tf.sigmoid(convRawProb);

            return tf.concat(new[] { predictedXYWH, predictedConf, predictedProb }, axis: -1);
        }

        static readonly int[] DefaultXYScale = { 1, 1, 1, };
        const float DefaultIntersectionOverUnionLossThreshold = 0.5f;
    }
}
