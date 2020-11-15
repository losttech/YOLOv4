namespace tensorflow {
    using System.Diagnostics;
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
        public bool LogDevicePlacement { get; set; }
        public bool GpuAllowGrowth { get; set; }

        public override int Run(string[] remainingArguments) {
            Trace.Listeners.Add(new ConsoleTraceListener(useErrorStream: true));

            tf.enable_eager_execution();

            tf.debugging.set_log_device_placement(this.LogDevicePlacement);

            if (this.GpuAllowGrowth) {
                dynamic config = config_pb2.ConfigProto.CreateInstance();
                config.gpu_options.allow_growth = true;
                tf.keras.backend.set_session(Session.NewDyn(config: config));
            }

            var dataset = new ObjectDetectionDataset(this.Annotations,
                classNames: this.ClassNames,
                strides: this.Strides,
                inputSize: this.InputSize,
                anchors: this.Anchors,
                anchorsPerScale: this.AnchorsPerScale,
                maxBBoxPerScale: this.MaxBBoxPerScale);
            var model = YOLO.CreateV4Trainable(dataset.InputSize, dataset.ClassNames.Length, dataset.Strides);
            // https://github.com/AlexeyAB/darknet/issues/1845
            var optimizer = new Adam(epsilon: 0.000001);
            YOLO.Train(model, optimizer, dataset, batchSize: this.BatchSize,
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
            this.HasOption("log-device-placement", "Enables TensorFlow device placement logging",
                (string onOff) => this.LogDevicePlacement = onOff == "on");
            this.HasOption("gpu-allow-growth", "Makes TensorFlow allocate GPU memory as needed (default: reserve all GPU memory)",
                (string onOff) => this.GpuAllowGrowth = onOff == "on");
        }
    }
}
