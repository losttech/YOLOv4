namespace tensorflow.keras.layers {
    using System.Collections.Generic;

    using LostTech.Gradient.ManualWrappers;
    public class FreezableBatchNormalization : BatchNormalization {
        Tensor ShouldTrain(bool training) => tf.logical_and(training, this.trainable);
        Tensor ShouldTrain(IGraphNodeBase? training)
            => tf.logical_and(training ?? tf.constant(false), this.trainable);

        public override Tensor call(IGraphNodeBase inputs, IGraphNodeBase? training = null)
            => base.call(inputs, this.ShouldTrain(training));
        public override Tensor call(IGraphNodeBase inputs, bool training)
            => base.call(inputs, this.ShouldTrain(training));
        public override Tensor call(IEnumerable<IGraphNodeBase> inputs, bool training)
            => base.call(inputs, this.ShouldTrain(training));
    }
}
