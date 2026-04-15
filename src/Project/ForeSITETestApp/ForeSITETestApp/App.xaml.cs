using System.Windows;
using System.Windows.Threading;

namespace ForeSITETestApp;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : Application
{
    private async void Application_Startup(object sender, StartupEventArgs e)
    {
        ShutdownMode = ShutdownMode.OnExplicitShutdown;

        var splash = new SplashWindow();
        splash.Show();

        void CloseSplash()
        {
            if (splash.IsVisible)
            {
                splash.Close();
            }
        }

        try
        {
            // Let the splash render before heavier startup work begins.
            await Dispatcher.Yield(DispatcherPriority.Render);
            await Task.Delay(900);

            var mainWindow = new MainWindow();
            MainWindow = mainWindow;
            mainWindow.ContentRendered += (_, __) => CloseSplash();
            mainWindow.Closed += (_, __) => CloseSplash();
            mainWindow.Show();

            // Fallback: even if ContentRendered is delayed, close splash after main window is shown.
            await Dispatcher.Yield(DispatcherPriority.Background);
            CloseSplash();
            ShutdownMode = ShutdownMode.OnMainWindowClose;
        }
        catch (Exception ex)
        {
            CloseSplash();
            MessageBox.Show(
                $"Application failed to start.\n{ex.Message}",
                "Startup Error",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            Shutdown(-1);
        }
        finally
        {
            CloseSplash();
        }
    }
}
