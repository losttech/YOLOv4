namespace tensorflow {
    using System;
    using System.Collections.Generic;

    using ManyConsole.CommandLineUtils;

    using numpy;

    using tensorflow.datasets.ObjectDetection;
    using tensorflow.keras.applications;
    using tensorflow.keras.models;

    class ToSavedModel : ConsoleCommand {
        public string WeigthsPath { get; set; }
        public string OutputPath { get; set; }
        public int InputSize { get; set; } = MS_COCO.InputSize;
        public int ClassCount { get; set; } = MS_COCO.ClassCount;
        public float ScoreThreshold { get; set; } = 0.2f;
        public int[] Strides { get; set; } = YOLOv4.Strides.ToArray();
        public ndarray<int> Anchors { get; set; } = YOLOv4.Anchors;
        public override int Run(string[] remainingArguments) {
            var trainable = YOLO.CreateV4Trainable(inputSize: this.InputSize,
                                                 classCount: this.ClassCount,
                                                 strides: this.Strides);
            trainable.load_weights(this.WeigthsPath);
            var output = YOLOv4.Output.Get(trainable);
            Tensor input = trainable.input_dyn;
            var savable = YOLO.CreateSaveable(inputSize: this.InputSize, input, output,
                                             classCount: this.ClassCount,
                                             strides: this.Strides,
                                             anchors: tf.constant(this.Anchors),
                                             xyScale: YOLOv4.XYScale,
                                             scoreThreshold: this.ScoreThreshold);
            savable.summary();
            savable.save(this.OutputPath, save_format: "tf", include_optimizer: false);
            return 0;
        }

        public ToSavedModel() {
            this.IsCommand("to-saved-model");
            this.HasRequiredOption("w|weights=", "Path to weights file (.index)",
                path => this.WeigthsPath = path);
            this.HasRequiredOption("o|output=", "Path to the output file",
                path => this.OutputPath = path);
            this.HasOption("t|score-threshold=", "Minimal score for detections",
                (float threshold) => this.ScoreThreshold = threshold);
        }
    }
}
