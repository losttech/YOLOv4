namespace DetectUI {
    using System;
    using System.Collections.Generic;
    using System.Data;
    using System.IO;
    using System.Linq;
    using System.Windows.Forms;

    using LostTech.Gradient;
    using LostTech.Gradient.Exceptions;
    using LostTech.TensorFlow;

    using SixLabors.Fonts;
    using SixLabors.ImageSharp;
    using SixLabors.ImageSharp.Drawing;
    using SixLabors.ImageSharp.Drawing.Processing;
    using SixLabors.ImageSharp.PixelFormats;
    using SixLabors.ImageSharp.Processing;

    using tensorflow;
    using tensorflow.core.protobuf.config_pb2;
    using tensorflow.datasets.ObjectDetection;
    using tensorflow.keras;
    using tensorflow.keras.applications;
    using tensorflow.keras.layers;
    using tensorflow.keras.models;

    public partial class YoloForm : Form {
        Model model;
        // dynamic infer;
        bool loaded;
        public YoloForm() {
            this.InitializeComponent();

            this.openPicDialog.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);

            GradientEngine.UseEnvironmentFromVariable();
            TensorFlowSetup.Instance.EnsureInitialized();

            // TODO: remove this after replacing tf.sigmoid in PostProcessBBBox
            tf.enable_eager_execution();
            tf.enable_v2_behavior();

            dynamic config = config_pb2.ConfigProto.CreateInstance();
            config.gpu_options.allow_growth = true;
            tf.keras.backend.set_session(Session.NewDyn(config: config));
        }

        private void openPic_Click(object sender, EventArgs e) {
            if (this.openPicDialog.ShowDialog(this) != DialogResult.OK)
                return;

            using var image = Image.Load<Rgb24>(this.openPicDialog.FileName);

            // TODO: use non-raw model. Raw is extremely slow
            ObjectDetectionResult[] detections = YOLO.DetectRaw(this.model,
                supportedSize: new Size(MS_COCO.InputSize, MS_COCO.InputSize),
                image: image);

            image.Mutate(context => {
                var font = SystemFonts.CreateFont("Arial", 16);
                var textColor = Color.White;
                var boxPen = new Pen(Color.White, width: 4);
                foreach(var detection in detections) {
                    string className = detection.Class < MS_COCO.ClassCount && detection.Class >= 0
                        ? MS_COCO.ClassNames[detection.Class] : "imaginary class";
                    string text = $"{className}: {detection.Score:P0}";
                    var box = Scale(detection.Box, image.Size());
                    context.DrawText(text, font, textColor, TopLeft(box));
                    var drawingBox = new RectangularPolygon(box);
                    context.Draw(boxPen, drawingBox);
                }
            });

            using var temp = new MemoryStream();
            image.SaveAsBmp(temp);
            temp.Position = 0;

            this.pictureBox.Image = new System.Drawing.Bitmap(temp);

            this.Text = string.Join(", ", detections.Select(d => MS_COCO.ClassNames[d.Class]));
        }

        static PointF TopLeft(RectangleF rect) => new PointF(rect.Left, rect.Top);
        static RectangleF Scale(RectangleF rect, SizeF size)
            => new RectangleF(x: rect.Left * size.Width, width: rect.Width * size.Width,
                              y: rect.Top * size.Height, height: rect.Height * size.Height);

        void LoadWeights() {
            while (!this.loaded) {
                string modelPath = Environment.GetEnvironmentVariable("DETECT_UI_WEIGHTS");
                if (modelPath is null) {
                    if (this.openWeightsDirDialog.ShowDialog(this) != DialogResult.OK)
                        continue;

                    modelPath = this.openWeightsDirDialog.SelectedPath;
                }

                try {
                    this.model = YOLO.CreateRaw(inputSize: MS_COCO.InputSize, classCount: MS_COCO.ClassCount);
                    this.model.summary();
                    this.model.load_weights(modelPath);
                    //this.model = tf.saved_model.load_v2(modelDir, tags: tf.saved_model.SERVING);
                    //this.infer = this.model.signatures["serving_default"];
                } catch (ValueError e) {
                    this.Text = e.Message;
                    continue;
                }
                this.loaded = true;
                this.Text = "YOLO " + modelPath;
            }

            this.openPic.Enabled = true;
        }

        private void YoloForm_Load(object sender, EventArgs e) {
            this.LoadWeights();
        }
    }
}
