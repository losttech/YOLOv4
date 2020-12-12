namespace tensorflow.keras.applications {
    using LostTech.Gradient;
    using LostTech.Gradient.BuiltIns;

    using tensorflow.keras.models;
    partial class YOLO {
        public static Model CreateRaw(int inputSize, int classCount) {
            Tensor input = tf.keras.Input(new TensorShape(inputSize, inputSize, 3));
            var featureMaps = YOLOv4.Apply(input, classCount: classCount);
            var featureMapTensors = new PythonList<Tensor> { featureMaps.SSBox, featureMaps.MBBox, featureMaps.LBBox };
            return new Model(new { inputs = input, outputs = featureMapTensors }.AsKwArgs());
        }
    }
}
