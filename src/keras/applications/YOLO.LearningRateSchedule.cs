namespace tensorflow.keras.applications {
    using System;
    using System.Collections.Generic;
    using System.ComponentModel;

    using LostTech.Gradient;
    using LostTech.Gradient.ManualWrappers;
    partial class YOLO {
        public class LearningRateSchedule : optimizers.schedules.LearningRateSchedule {
            readonly Tensor totalSteps, warmupSteps, initialLR, finalLR;

            public long TotalSteps { get; }
            public long WarmupSteps { get; }
            public float InitialLearningRate { get; }
            public float FinalLearningRate { get; }

            public LearningRateSchedule(long totalSteps, long warmupSteps,
                float initialLearningRate = 1e-3f,
                float finalLearningRate = 1e-6f) {
                if (totalSteps <= 0) throw new ArgumentOutOfRangeException(nameof(totalSteps));
                if (warmupSteps <= 0) throw new ArgumentOutOfRangeException(nameof(warmupSteps));
                if (!GoodLearningRate(initialLearningRate))
                    throw new ArgumentOutOfRangeException(nameof(initialLearningRate));
                if (!GoodLearningRate(finalLearningRate))
                    throw new ArgumentOutOfRangeException(nameof(finalLearningRate));

                this.TotalSteps = totalSteps;
                this.WarmupSteps = warmupSteps;
                this.InitialLearningRate = initialLearningRate;
                this.FinalLearningRate = finalLearningRate;

                using var _ = Python.Runtime.Py.GIL();
                this.totalSteps = tf.constant_scalar(totalSteps);
                this.warmupSteps = tf.constant_scalar(warmupSteps);
                this.initialLR = tf.constant_scalar(initialLearningRate);
                this.finalLR = tf.constant_scalar(finalLearningRate);
            }

            public override IDictionary<string, object> get_config() {
                throw new NotImplementedException();
            }

            public Tensor Get(IGraphNodeBase step) => this.__call__(step);

            [EditorBrowsable(EditorBrowsableState.Advanced)]
            public override dynamic __call__(IGraphNodeBase step)
                => tf.cond(step < this.warmupSteps,
                    PythonFunctionContainer.Of<Tensor>(() => tf.cast(step / this.warmupSteps, tf.float32) * this.initialLR),
                    PythonFunctionContainer.Of<Tensor>(() => this.finalLR
                        + 0.5f * (this.initialLR - this.finalLR)
                            * (1 + tf.cos(
                                    tf.cast(step - this.warmupSteps, tf.float32)
                                    / tf.cast(this.totalSteps - this.warmupSteps, tf.float32)
                                    * Math.PI)))
                );

            static bool GoodLearningRate(float lr)
                => lr > 0 && !float.IsPositiveInfinity(lr);

            public override dynamic __call___dyn(object step) => throw new NotImplementedException();
            public override dynamic get_config_dyn() => throw new NotImplementedException();
        }
    }
}
