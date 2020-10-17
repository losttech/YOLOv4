namespace tensorflow.image {
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;

    using LostTech.Gradient;

    using numpy;

    using tensorflow.data;

    static class ImageTools {
        /// <summary>
        /// Returns bytes of the image in HWC order
        /// </summary>
        public unsafe static byte[,,] LoadRGB8(string filePath) {
            using var bitmap = new Bitmap(filePath);
            int channels = Image.IsAlphaPixelFormat(bitmap.PixelFormat) ? 4 : 3;
            byte[,,] result = new byte[bitmap.Height, bitmap.Width, channels];
            var lockFormat = channels == 3 ? PixelFormat.Format24bppRgb : PixelFormat.Format32bppArgb;

            var data = bitmap.LockBits(new Rectangle { Width = bitmap.Width, Height = bitmap.Height },
                                       ImageLockMode.ReadOnly, lockFormat);
            try {
                for (int y = 0; y < bitmap.Height; y++) {
                    fixed (byte* targetStride = &result[y, 0, 0]) {
                        byte* sourceStride = (byte*)data.Scan0 + y * data.Stride;
                        var sourceSpan = new ReadOnlySpan<byte>(sourceStride, bitmap.Width * channels);
                        var targetSpan = new Span<byte>(targetStride, bitmap.Width * channels);
                        sourceSpan.CopyTo(targetSpan);
                    }
                }
            } finally {
                bitmap.UnlockBits(data);
            }

            return result;
        }

        public static ObjectDetectionDataset.Entry<float> YoloPreprocess(ObjectDetectionDataset.Entry<float> entry, Size targetSize) {
            if (entry.Image is null) throw new ArgumentNullException(nameof(image));

            int h = entry.Image.shape.Item1, w = entry.Image.shape.Item2;
            float scale = Math.Min(targetSize.Width * 1f / w, targetSize.Height *1f / h);
            int newW = (int)(scale * w), newH = (int)(scale * h);

            var resized = Resize(entry.Image, width: newW, height: newH);

            var padded = (ndarray<float>)np.full(shape: new[] { targetSize.Height, targetSize.Width, 3 },
                                                 fill_value: 128f, dtype: dtype.GetClass<float>());
            int dw = (targetSize.Width - newW) / 2, dh = (targetSize.Height - newH) / 2;
            padded[dh..(newH + dh), dw..(newW + dw)] = resized;
            padded = (ndarray<float>)(padded / 255f);

            if (entry.BoundingBoxes != null) {
                var horIndex = (.., new[] { 0, 2 });
                var vertIndex = (.., new[] { 1, 3 });
                entry.BoundingBoxes.__setitem__(horIndex,
                    value: entry.BoundingBoxes.__getitem__(horIndex) * scale + dw);
                entry.BoundingBoxes.__setitem__(vertIndex,
                    value: entry.BoundingBoxes.__getitem__(vertIndex) * scale + dh);
            }

            entry.Image = padded;

            return entry;
        }

        static ndarray<float> Resize(ndarray<float> image, int width, int height) {
            var graphs = new Dictionary<(int, int), Graph>();
            if (!graphs.TryGetValue((width, height), out var graph)) {
                graphs[(width, height)] = graph = new Graph();
                var def = graph.as_default_dyn();
                def.__enter__();
                var @in = tf.placeholder(tf.float32, new TensorShape(null, null, 3), name: "input");
                var @out = tf.image.resize(@in, new object[] { height, width }, ResizeMethod.BILINEAR, name: "output");
                def.__exit__(null, null, null);
            }
            var session = new Session(graph: graph);
            using var _ = session.StartUsing();
            var runResult = session.run(graph.get_tensor_by_name("output/Squeeze:0"), new Dictionary<object, object> {
                [graph.get_tensor_by_name("input:0")] = image
            });
            return (ndarray<float>)runResult;
        }
    }
}
