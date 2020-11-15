namespace tensorflow.image {
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;

    using LostTech.Gradient;

    using numpy;

    using SixLabors.ImageSharp;
    using SixLabors.ImageSharp.Processing;

    using tensorflow.data;

    using Size = SixLabors.ImageSharp.Size;

    static class ImageTools {
        /// <summary>
        /// Returns bytes of the image in HWC order
        /// </summary>
        public unsafe static byte[,,] LoadRGB8(string filePath) {
            using var bitmap = new Bitmap(filePath);
            int channels = System.Drawing.Image.IsAlphaPixelFormat(bitmap.PixelFormat) ? 4 : 3;
            byte[,,] result = new byte[bitmap.Height, bitmap.Width, channels];
            var lockFormat = channels == 3 ? PixelFormat.Format24bppRgb : PixelFormat.Format32bppArgb;

            var data = bitmap.LockBits(new System.Drawing.Rectangle { Width = bitmap.Width, Height = bitmap.Height },
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

        public static ObjectDetectionDataset.Entry<float> YoloPreprocess(ObjectDetectionDataset.ClrEntry entry, Size targetSize) {
            if (entry.Image is null) throw new ArgumentNullException(nameof(image));

            int h = entry.Image.Height, w = entry.Image.Width;
            float scale = Math.Min(targetSize.Width * 1f / w, targetSize.Height *1f / h);
            int newW = (int)(scale * w), newH = (int)(scale * h);

            Resize(entry.Image, width: newW, height: newH);

            var padded = (ndarray<float>)np.full(shape: new[] { targetSize.Height, targetSize.Width, 3 },
                                                 fill_value: 128f, dtype: dtype.GetClass<float>());
            int dw = (targetSize.Width - newW) / 2, dh = (targetSize.Height - newH) / 2;
            padded[dh..(newH + dh), dw..(newW + dw)] = entry.ToNumPyEntry().Image;
            padded /= 255f;

            if (entry.BoundingBoxes != null) {
                var horIndex = (.., new[] { 0, 2 });
                var vertIndex = (.., new[] { 1, 3 });
                entry.BoundingBoxes[horIndex] = (entry.BoundingBoxes[horIndex] * scale).astype(np.int32_fn).AsArray<int>() + dw;
                entry.BoundingBoxes[vertIndex] = (entry.BoundingBoxes[vertIndex] * scale).astype(np.int32_fn).AsArray<int>() + dh;
            }

            return new ObjectDetectionDataset.Entry<float> {
                Image = padded,
                BoundingBoxes = entry.BoundingBoxes,
            };
        }

        static void Resize(SixLabors.ImageSharp.Image image, int width, int height) {
            image.Mutate(x => x.Resize(width, height, KnownResamplers.Box));
        }
    }
}
