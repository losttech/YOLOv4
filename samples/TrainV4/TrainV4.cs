namespace tensorflow {
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;

    using LostTech.Gradient;

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
        public ndarray<float> Anchors { get; set; } = YOLOv4.Anchors.AsType<float>();
        public int AnchorsPerScale { get; set; } = YOLOv4.AnchorsPerScale;
        public int[] Strides { get; set; } = YOLOv4.Strides.ToArray();
        public bool LogDevicePlacement { get; set; }
        public bool GpuAllowGrowth { get; set; }
        public bool ModelSummary { get; set; }
        public bool TestRun { get; set; }
        public bool Benchmark { get; set; }
        public int FirstStageEpochs { get; set; } = 20;
        public int SecondStageEpochs { get; set; } = 30;
        public int WarmupEpochs { get; set; } = 2;
        public string LogDir { get; set; }
        public string? WeightsPath { get; set; }

        public override int Run(string[] remainingArguments) {
            Trace.Listeners.Add(new ConsoleTraceListener(useErrorStream: true));

            tf.debugging.set_log_device_placement(this.LogDevicePlacement);

            if (this.GpuAllowGrowth) {
                dynamic config = config_pb2.ConfigProto.CreateInstance();
                config.gpu_options.allow_growth = true;
                tf.keras.backend.set_session(Session.NewDyn(config: config));
            }

            if (this.TestRun)
                this.Annotations = this.Annotations.Take(this.BatchSize*3).ToArray();

            var dataset = new ObjectDetectionDataset(this.Annotations,
                classNames: this.ClassNames,
                strides: this.Strides,
                inputSize: this.InputSize,
                anchors: this.Anchors,
                anchorsPerScale: this.AnchorsPerScale,
                maxBBoxPerScale: this.MaxBBoxPerScale);
            var model = YOLO.CreateV4Trainable(dataset.InputSize, dataset.ClassNames.Length, dataset.Strides);

            var learningRateSchedule = new YOLO.LearningRateSchedule(
                totalSteps: (long)(this.FirstStageEpochs + this.SecondStageEpochs) * dataset.BatchCount(this.BatchSize),
                warmupSteps: this.WarmupEpochs * dataset.BatchCount(this.BatchSize));
            // https://github.com/AlexeyAB/darknet/issues/1845
            var optimizer = new Adam(learning_rate: learningRateSchedule, epsilon: 0.000001);
            if (this.ModelSummary)
                model.summary();
            if (this.WeightsPath != null)
                model.load_weights(this.WeightsPath);

            var callbacks = new List<ICallback> {
                new LearningRateLogger(),
                new TensorBoard(log_dir: this.LogDir, batch_size: this.BatchSize, profile_batch: 4),
            };
            if (!this.Benchmark && !this.TestRun)
                callbacks.Add(new ModelCheckpoint("yoloV4.weights.{epoch:02d}", save_weights_only: true));

            YOLO.TrainGenerator(model, optimizer, dataset, batchSize: this.BatchSize,
                       firstStageEpochs: this.FirstStageEpochs,
                       secondStageEpochs: this.SecondStageEpochs,
                       callbacks: callbacks);

            if (!this.Benchmark && !this.TestRun)
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
            this.HasOption("model-summary", "Print model summary before training",
                (string onOff) => this.ModelSummary = onOff == "on");
            this.HasOption("log-dir=", "Write training logs to the specified directory",
                dir => {
                    dir = Path.GetFullPath(dir);
                    Directory.CreateDirectory(dir);
                    this.LogDir = dir;
                });
            this.HasOption("transfer-epochs=", "Number of epochs to run to adapt before fine-tuning",
                (int epochs) => this.FirstStageEpochs = epochs);
            this.HasOption("training-epochs=", "Number of epochs to run training/fine-tuning for",
                (int epochs) => this.SecondStageEpochs = epochs);
            this.HasOption("test-run", "Only does 1 batch per epoch instead of the entire dataset",
                (string onOff) => this.TestRun = onOff == "on");
            this.HasOption("weights=", "Path to pretrained model weights",
                (string path) => this.WeightsPath = path);
            this.HasOption("benchmark", "Run 1 epoch without training and output losses",
                (string onOff) => this.Benchmark = onOff == "on");
        }
    }
}
