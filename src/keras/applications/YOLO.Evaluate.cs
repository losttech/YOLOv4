namespace tensorflow.keras.applications {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using LostTech.Gradient;
    using LostTech.Gradient.BuiltIns;

    using numpy;

    using SixLabors.ImageSharp;
    using SixLabors.ImageSharp.PixelFormats;

    using tensorflow.data;
    using tensorflow.datasets.ObjectDetection;
    using tensorflow.image;
    using tensorflow.keras.models;
    static partial class YOLO {
        static Tensor DecodeEval(Tensor convOut, int classCount) {
            Tensor batchSize = tf.shape(convOut)[0];
            Tensor outputSize = tf.shape(convOut)[1];

            convOut = tf.reshape_dyn(convOut, new object[] { batchSize, outputSize, outputSize, 3, 5 + classCount });
            Tensor[] raws = tf.split(convOut, new[] { 4, 1, classCount }, axis: -1);
            var (convRawXYWH, convRawConf, convRawProb) = raws;

            Tensor predConf = tf.sigmoid(convRawConf);
            Tensor predProb = tf.sigmoid(convRawProb);

            return tf.concat(new[] { convRawXYWH, predConf, predProb }, axis: -1);
        }

        public static Model CreateV4EvalOnly(int inputSize, int classCount) {
            if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
            if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));

            Tensor input = tf.keras.Input(new TensorShape(inputSize, inputSize, 3));
            var featureMaps = YOLOv4.Apply(input, classCount: classCount);

            var bboxTensors = new PythonList<Tensor>();
            foreach (var featureMap in new[] { featureMaps.SSBox, featureMaps.MBBox, featureMaps.LBBox }) {
                var bbox = DecodeEval(featureMap, classCount: classCount);
                bboxTensors.Add(bbox);
            }
            return new Model(new { inputs = input, outputs = bboxTensors }.AsKwArgs());
        }

        public static ObjectDetectionResult[] Detect(dynamic detector, Size supportedSize, Image<Rgb24> image) {
            if (detector is null) throw new ArgumentNullException(nameof(detector));
            if (image is null) throw new ArgumentNullException(nameof(image));

            var input = ImageTools.YoloPreprocess(new ObjectDetectionDataset.ClrEntry {
                Image = image.Clone(),
            }, supportedSize);
            using var _ = Python.Runtime.Py.GIL();
            var images = input.Image[np.newaxis, np.rest_of_the_axes].AsArray();

            IList<ndarray<float>> prediction = detector(tf.constant(images));
            //var bbBox = PostProcessBBBox(prediction, YOLOv4.Anchors, YOLOv4.Strides, YOLOv4.XYScale);
            //dynamic numpy = PythonModuleContainer.Get<np>();
            //ndarray<float> outputs = numpy.concatenate(bbBox, axis: 0);
            ////tf.image.combined_non_max_suppression()
            //var processed = PostProcessBoxes(outputs[np.newaxis].AsArray(), image.Size(), supportedSize.Width, 0.25f);
            return Array.Empty<ObjectDetectionResult>();
        }

        public static ObjectDetectionResult[] DetectRaw(Model rawDetector, Size supportedSize, Image<Rgb24> image) {
            if (rawDetector is null) throw new ArgumentNullException(nameof(rawDetector));
            if (image is null) throw new ArgumentNullException(nameof(image));

            var input = ImageTools.YoloPreprocess(new ObjectDetectionDataset.ClrEntry {
                Image = image.Clone(),
            }, supportedSize);
            var images = input.Image[np.newaxis, np.rest_of_the_axes].AsArray();

            IList<Tensor<float>> prediction = rawDetector.__call___dyn(images);
            Debug.Assert(prediction.Count == 3);
            var output = new YOLOv4.Output {
                SSBox = prediction[0],
                MBBox = prediction[1],
                LBBox = prediction[2],
            };

            var pred = ProcessPrediction(inputSize: MS_COCO.InputSize, output,
                                         classCount: MS_COCO.ClassCount,
                                         strides: YOLOv4.Strides,
                                         anchors: tf.constant(YOLOv4.Anchors),
                                         xyScale: YOLOv4.XYScale,
                                         scoreThreshold: 0.2f);

            var boxes = pred[.., .., 0..4];
            var conf = pred[.., .., 4..];

            var batchSize = tf.shape(boxes)[0];

            var suppression = tf.image.combined_non_max_suppression(
                boxes: tf.reshape_dyn(boxes, new object[] { batchSize, -1, 1, 4 }),
                scores: tf.reshape_dyn(conf, new object[] { batchSize, -1, tf.shape(conf)[^1] }),
                max_output_size_per_class: tf.constant(50),
                max_total_size: tf.constant(50),
                iou_threshold: 0.45f,
                score_threshold: 0.25f
                );

            ndarray<float> boxs = suppression[0].numpy();
            ndarray<float> scores = suppression[1].numpy();
            ndarray<float> classes = suppression[2].numpy();
            ndarray<int> detections = suppression[3].numpy();

            if (detections[0].AsScalar() == 0) {
                Debug.WriteLine("nothing detected");
                return Array.Empty<ObjectDetectionResult>();
            }

            return Array.Empty<ObjectDetectionResult>();
        }

        public static ObjectDetectionResult[] Detect(Model detector, Size supportedSize, Image<Rgb24> image) {
            if (detector is null) throw new ArgumentNullException(nameof(detector));
            if (image is null) throw new ArgumentNullException(nameof(image));

            var input = ImageTools.YoloPreprocess(new ObjectDetectionDataset.ClrEntry {
                Image = image.Clone(),
            }, supportedSize);
            using var _ = Python.Runtime.Py.GIL();
            var images = input.Image[np.newaxis, np.rest_of_the_axes].AsArray();

            IList<ndarray<float>> prediction = detector.__call___dyn(images);
            //var bbBox = PostProcessBBBox(prediction, YOLOv4.Anchors, YOLOv4.Strides, YOLOv4.XYScale);
            //dynamic numpy = PythonModuleContainer.Get<np>();
            //ndarray<float> outputs = numpy.concatenate(bbBox, axis: 0);
            ////tf.image.combined_non_max_suppression()
            //var processed = PostProcessBoxes(outputs[np.newaxis].AsArray(), image.Size(), supportedSize.Width, 0.25f);
            return Array.Empty<ObjectDetectionResult>();
        }


        static ndarray<float> PostProcessBBBox(IEnumerable<ndarray<float>> predictions,
                                     ndarray<int> anchors,
                                     ReadOnlySpan<int> strides,
                                     ReadOnlySpan<float> xyScale) {
            foreach(var (scaleIndex, pred) in Tools.Enumerate(predictions.ToArray())) {
                var convShape = pred.shape;
                int outputSize = convShape.Item2;
                var convRawDxDy = pred[.., .., .., .., 0..2];
                var convRawDwDh = pred[.., .., .., .., 2..4];

                dynamic numpy = PythonModuleContainer.Get<np>();
                var sizeRange = Enumerable.Range(0, outputSize).ToNumPyArray();
                PythonList<ndarray<int>> tempGrid = numpy.meshgrid(sizeRange, sizeRange);
                ndarray<int> xyGrid = np.expand_dims(
                    np.stack(tempGrid, axis: -1),
                    axis: 2).AsArray<int>(); // [gx, gy, 1, 2]

                xyGrid = numpy.tile(np.expand_dims(xyGrid, axis: 0), new[] { 1, 1, 1, 3, 1 });
                var xyGridFloat = xyGrid.AsType<float>();

                var predXY = ((tf.sigmoid_dyn(convRawDxDy).numpy() * xyScale[scaleIndex]) - 0.5f * (xyScale[scaleIndex] - 1) + xyGridFloat) * strides[scaleIndex];
                ndarray<double> predWH = numpy.exp(convRawDwDh) * anchors[scaleIndex];

                pred[.., .., .., .., 0..2] = predXY;
                pred[.., .., .., .., 2..4] = predWH.AsType<float>();
            }

            var reshapedPredictions = predictions.Select(
                x => x.reshape(new[] { -1, (int)x.shape.Item5 }).AsArray<float>());
            return np.concatenate(reshapedPredictions, axis: 0);
        }

        static ndarray<float> PostProcessBoxes(ndarray<float> predictions, Size originalSize, int inputSize, float scoreThreshold) {
            var predXYWH = predictions[.., 0..4];
            var predConf = predictions[.., 4];
            var predProb = predictions[.., 5..];

            // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
            var predCoor = np.concatenate(new[]{
                    predXYWH[.., ..2] - predXYWH[..,2..]*0.5f,
                    predXYWH[.., ..2] + predXYWH[..,2..]*0.5f,
                }, axis: -1);

            // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
            var (h, w) = (originalSize.Height, originalSize.Width);
            float resizeRatio = Math.Min(inputSize * 1f / w, inputSize * 1f / h);

            float dw = (inputSize - resizeRatio * w) / 2;
            float dh = (inputSize - resizeRatio * h) / 2;

            for (int i = 0; i < 2; i++) {
                predCoor[.., i * 2] = (predCoor[.., i * 2] - dw) / resizeRatio;
                predCoor[.., i * 2 + 1] = (predCoor[.., i * 2 + 1] - dh) / resizeRatio;
            }

            // (3) clip some boxes those are out of range
            predCoor = np.concatenate(new[] {
                np.maximum(predCoor[.., ..2], np.zeros<float>(2)),
                np.maximum(predCoor[.., 2..], new []{w-1f, h-1f}.ToNumPyArray()),
            }, axis: -1);
            dynamic numpy = PythonModuleContainer.Get<np>();
            ndarray<bool> invalidMask = numpy.logical_or(predCoor[.., 0] > predCoor[.., 2], predCoor[.., 1] > predCoor[.., 3]);
            predCoor[invalidMask] = 0;

            // (4) discard some invalid boxes
            ndarray<float> bboxesScale = numpy.sqrt(numpy.multiply.reduce(
                predCoor[.., 2..4] - predCoor[.., 0..2],
                axis: -1));
            ndarray<bool> scaleMask = numpy.logical_and(0 < bboxesScale, bboxesScale < float.PositiveInfinity);

            // (5) discard some boxes with low scores
            var classes = predProb.argmax(axis: -1).AsArray<long>();
            ndarray<float> scores = predConf * predProb[numpy.arange(predCoor.shape.Item1), classes];
            ndarray<bool>? scoreMask = scores > scoreThreshold;
            ndarray<bool> mask = numpy.logical_and(scaleMask, scoreMask);

            var coords = predCoor[mask].AsArray();
            scores = scores[mask].AsArray();
            classes = classes[mask].AsArray();

            return np.concatenate(new [] {
                coords,
                scores[.., np.newaxis],
                classes[.., np.newaxis].AsArray().AsType<float>(),
            }, axis: -1);
        }
    }
}
