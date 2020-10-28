namespace tensorflow.keras.applications {
    using System;

    using LostTech.Gradient;
    using LostTech.Gradient.BuiltIns;

    using numpy;

    using SixLabors.ImageSharp;
    using SixLabors.ImageSharp.PixelFormats;

    using tensorflow.data;
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
                bboxTensors.Add(featureMap);
                bboxTensors.Add(bbox);
            }
            return new Model(new { inputs = input, outputs = bboxTensors }.AsKwArgs());
        }

        public static ObjectDetectionResult[] Detect(Model detector, Size supportedSize, Image<Rgb24> image) {
            if (detector is null) throw new ArgumentNullException(nameof(detector));
            if (image is null) throw new ArgumentNullException(nameof(image));

            var input = ImageTools.YoloPreprocess(new ObjectDetectionDataset.ClrEntry {
                Image = image,
            }, supportedSize);
            input.Image = input.Image[np.newaxis, np.rest_of_the_axes].AsArray();

            var bbox = detector.predict(input.Image);
            return Array.Empty<ObjectDetectionResult>();
        }
    }
}
