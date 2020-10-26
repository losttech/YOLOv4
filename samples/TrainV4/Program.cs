namespace tensorflow {
    using System;

    using LostTech.Gradient;
    using LostTech.TensorFlow;

    using ManyConsole.CommandLineUtils;

    class Program {
        static int Main(string[] args) {
            Console.Title = "YOLOv4";
            GradientEngine.UseEnvironmentFromVariable();
            TensorFlowSetup.Instance.EnsureInitialized();

            return ConsoleCommandDispatcher.DispatchCommand(
                ConsoleCommandDispatcher.FindCommandsInSameAssemblyAs(typeof(Program)),
                args, Console.Out);
        }
    }
}
