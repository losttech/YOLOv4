namespace tensorflow {
    using System.Linq;

    using ManyConsole.CommandLineUtils;

    using numpy;

    using tensorflow.core.protobuf.config_pb2;
    using tensorflow.data;
    using tensorflow.datasets.ObjectDetection;
    using tensorflow.keras.applications;
    using tensorflow.keras.callbacks;
    using tensorflow.keras.models;
    using tensorflow.keras.optimizers;

    class TrainV4 : ConsoleCommand {
        public string[] Annotations { get; set; }
        public string[] ClassNames { get; set; }
        public int InputSize { get; set; } = MS_COCO.InputSize;
        public int MaxBBoxPerScale { get; set; } = 150;
        public int BatchSize { get; set; } = 2;
        public ndarray<float> Anchors { get; set; } = ObjectDetectionDataset.ParseAnchors(YOLOv4.Anchors.ToArray());
        public int AnchorsPerScale { get; set; } = YOLOv4.AnchorsPerScale;
        public int[] Strides { get; set; } = YOLOv4.Strides.ToArray();

        public override int Run(string[] remainingArguments) {
            tf.enable_eager_execution();

            dynamic config = config_pb2.ConfigProto.CreateInstance();
            config.gpu_options.allow_growth = true;
            tf.keras.backend.set_session(Session.NewDyn(config: config));

            tf.keras.mixed_precision.experimental.set_policy("infer_with_float32_vars");

            var dataset = new ObjectDetectionDataset(this.Annotations,
                classNames: this.ClassNames,
                strides: this.Strides,
                inputSize: this.InputSize,
                anchors: this.Anchors,
                anchorsPerScale: this.AnchorsPerScale,
                maxBBoxPerScale: this.MaxBBoxPerScale);
            var optimizer = new Adam();
            var model = YOLO.Train(optimizer, dataset, batchSize: this.BatchSize,
                                   callbacks: new ICallback[] {
                                       new BaseLogger(),
                                       new TrainingLogger(),
                                   });
            model.save_weights("yoloV4.weights-trained");
            // the following does not work due to the need to name layers properly
            // https://stackoverflow.com/questions/61402903/unable-to-create-group-name-already-exists
            // model.save("yoloV4-trained");
            return 0;
        }

        public TrainV4() {
            this.IsCommand("trainV4");
            this.HasRequiredOption("a|annotations=", "Path to MS COCO-compatible annotations file",
                filePath => this.Annotations = Tools.NonEmptyLines(filePath));
            this.HasRequiredOption("c|class-names=",
                "Path to MS COCO-compatible .names file listing all object classes",
                filePath => this.ClassNames = Tools.NonEmptyLines(filePath));
            this.HasOption("batch-size=", "Batch size during training",
                (int size) => this.BatchSize = size);
        }
    }
}
