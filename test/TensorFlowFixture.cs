namespace tensorflow.keras {
    using LostTech.Gradient;
    using LostTech.TensorFlow;

    using Xunit;

    public class TensorFlowFixture {
        public TensorFlowFixture() {
            TensorFlowSetup.Instance.EnsureInitialized();
        }
    }

    // see https://stackoverflow.com/questions/12976319/xunit-net-global-setup-teardown
    [CollectionDefinition("TF-dependent")]
    public class TensorFlowCollection: ICollectionFixture<TensorFlowFixture> { }
}
