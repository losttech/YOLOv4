namespace tensorflow.keras.layers {
    using System.Collections.Generic;

    using LostTech.Gradient.ManualWrappers;
    public class FreezableBatchNormalization : BatchNormalization {
        static readonly Tensor @false = tf.constant(false);

        IGraphNodeBase? ShouldTrain(IGraphNodeBase? training)
            => this.trainable ? training : @false;

        public override Tensor call(IGraphNodeBase inputs, IGraphNodeBase? training = null)
            => base.call(inputs, this.ShouldTrain(training));
        public override Tensor call(IGraphNodeBase inputs, bool training)
            => base.call(inputs, this.trainable && training);
        public override Tensor call(IEnumerable<IGraphNodeBase> inputs, bool training)
            => base.call(inputs, this.trainable && training);
    }
}
