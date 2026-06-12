using Microsoft.Win32;
using OpenCvWindowTool;
using System;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using PointF = System.Drawing.PointF;

namespace OpenCvWindowToolWpfDemo
{
    public partial class MainWindow : Window
    {
        private readonly OpenCvImageViewer viewer;
        private LineDetectionResult latestResult;
        private LineScanDirection currentDirection = LineScanDirection.LeftToRight;
        private OperatorModule currentModule = OperatorModule.Input;
        private bool initializing;

        public MainWindow()
        {
            initializing = true;
            InitializeComponent();
            viewer = new OpenCvImageViewer { DisplayToolBar = false };
            viewer.RoiChanged += Viewer_RoiChanged;
            viewer.RoiEditCompleted += Viewer_RoiEditCompleted;
            viewer.SelectedRoiChanged += Viewer_SelectedRoiChanged;
            FormsHost.Child = viewer;
            InitLineDetectionControls();
            ShowModule(OperatorModule.Input);
            initializing = false;
            RefreshLineDisplay(false);
        }

        private void InitLineDetectionControls()
        {
            PolarityComboBox.ItemsSource = new[]
            {
                new ComboOption<LineEdgePolarity>("从黑到白", LineEdgePolarity.Positive),
                new ComboOption<LineEdgePolarity>("从白到黑", LineEdgePolarity.Negative),
                new ComboOption<LineEdgePolarity>("全部", LineEdgePolarity.Any)
            };
            PolarityComboBox.SelectedIndex = 2;

            StrengthTypeComboBox.ItemsSource = new[]
            {
                new ComboOption<LineEdgeStrengthType>("一维梯度", LineEdgeStrengthType.Gradient1D),
                new ComboOption<LineEdgeStrengthType>("Sobel梯度", LineEdgeStrengthType.Sobel)
            };
            StrengthTypeComboBox.SelectedIndex = 0;

            SelectionModeComboBox.ItemsSource = new[]
            {
                new ComboOption<LineSelectionMode>("第一条", LineSelectionMode.First),
                new ComboOption<LineSelectionMode>("最后一条", LineSelectionMode.Last),
                new ComboOption<LineSelectionMode>("最强边", LineSelectionMode.Strongest)
            };
            SelectionModeComboBox.SelectedIndex = 2;

            FitModeComboBox.ItemsSource = new[]
            {
                new ComboOption<LineFitMode>("鲁棒拟合", LineFitMode.Robust),
                new ComboOption<LineFitMode>("最小二乘拟合", LineFitMode.LeastSquares)
            };
            FitModeComboBox.SelectedIndex = 0;
            UpdateDirectionButtons();
        }

        private void InputModule_Click(object sender, RoutedEventArgs e)
        {
            ShowModule(OperatorModule.Input);
            RefreshLineDisplay(false);
        }

        private void RoiModule_Click(object sender, RoutedEventArgs e)
        {
            ShowModule(OperatorModule.Roi);
            RefreshLineDisplay(false);
        }

        private void ParamsModule_Click(object sender, RoutedEventArgs e)
        {
            ShowModule(OperatorModule.Params);
            RefreshLineDisplay(true);
        }

        private void ResultModule_Click(object sender, RoutedEventArgs e)
        {
            ShowModule(OperatorModule.Result);
            RefreshLineDisplay(false);
        }

        private void ShowModule(OperatorModule module)
        {
            currentModule = module;
            InputPanel.Visibility = module == OperatorModule.Input ? Visibility.Visible : Visibility.Collapsed;
            RoiPanel.Visibility = module == OperatorModule.Roi ? Visibility.Visible : Visibility.Collapsed;
            ParamsPanel.Visibility = module == OperatorModule.Params ? Visibility.Visible : Visibility.Collapsed;
            ResultPanel.Visibility = module == OperatorModule.Result ? Visibility.Visible : Visibility.Collapsed;
            MarkModuleButton(InputModuleButton, module == OperatorModule.Input);
            MarkModuleButton(RoiModuleButton, module == OperatorModule.Roi);
            MarkModuleButton(ParamsModuleButton, module == OperatorModule.Params);
            MarkModuleButton(ResultModuleButton, module == OperatorModule.Result);
        }

        private static void MarkModuleButton(Button button, bool selected)
        {
            button.Background = selected ? new SolidColorBrush(ColorFromRgb(66, 133, 244)) : new SolidColorBrush(ColorFromRgb(48, 54, 62));
            button.Foreground = Brushes.White;
            button.BorderBrush = selected ? new SolidColorBrush(ColorFromRgb(66, 133, 244)) : new SolidColorBrush(ColorFromRgb(48, 54, 62));
        }

        private static Color ColorFromRgb(byte r, byte g, byte b)
        {
            return Color.FromRgb(r, g, b);
        }

        private void OpenImage_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog
            {
                Filter = "图像文件|*.bmp;*.jpg;*.jpeg;*.png;*.tif;*.tiff|所有文件|*.*"
            };

            if (dialog.ShowDialog(this) == true)
            {
                viewer.LoadImage(dialog.FileName);
                ImageStatusTextBlock.Text = dialog.FileName;
                latestResult = null;
                RefreshLineDisplay(false);
            }
        }

        private void CreateRectangleRoi_Click(object sender, RoutedEventArgs e)
        {
            viewer.StartCreateRoi(RoiShape.Rectangle);
            RoiStatusTextBlock.Text = "正在创建矩形检测ROI";
        }

        private void CreateRotatedRectangleRoi_Click(object sender, RoutedEventArgs e)
        {
            viewer.StartCreateRoi(RoiShape.RotatedRectangle);
            RoiStatusTextBlock.Text = "正在创建带角度矩形ROI";
        }

        private void ClearRoi_Click(object sender, RoutedEventArgs e)
        {
            viewer.ClearRois();
            viewer.ClearLineDetectionPreview();
            viewer.ClearLineDetectionResult();
            latestResult = null;
            RoiStatusTextBlock.Text = "未创建ROI";
            UpdateResultViews();
        }

        private void DirectionButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender == LeftToRightButton) currentDirection = LineScanDirection.LeftToRight;
            else if (sender == TopToBottomButton) currentDirection = LineScanDirection.TopToBottom;
            else if (sender == BottomToTopButton) currentDirection = LineScanDirection.BottomToTop;
            else if (sender == RightToLeftButton) currentDirection = LineScanDirection.RightToLeft;
            UpdateDirectionButtons();
            RefreshLineDisplay(true);
        }

        private void Parameter_Changed(object sender, RoutedEventArgs e)
        {
            if (initializing) return;
            RefreshLineDisplay(true);
        }

        private void Viewer_RoiChanged(object sender, RoiEventArgs e)
        {
            RoiStatusTextBlock.Text = viewer.SelectedRoi == null ? "未选择ROI" : viewer.SelectedRoi.Name;
            if (currentModule == OperatorModule.Params)
            {
                RefreshLineDisplay(false);
            }
        }

        private void Viewer_RoiEditCompleted(object sender, RoiEventArgs e)
        {
            RoiStatusTextBlock.Text = viewer.SelectedRoi == null ? "未选择ROI" : viewer.SelectedRoi.Name;
            if (currentModule == OperatorModule.Params)
            {
                RefreshLineDisplay(true);
            }
        }

        private void Viewer_SelectedRoiChanged(object sender, EventArgs e)
        {
            RoiStatusTextBlock.Text = viewer.SelectedRoi == null ? "未选择ROI" : viewer.SelectedRoi.Name;
            if (currentModule == OperatorModule.Params)
            {
                RefreshLineDisplay(true);
            }
        }

        private void RefreshLineDisplay(bool runDetection)
        {
            if (initializing || viewer == null) return;
            if (currentModule != OperatorModule.Params)
            {
                ResetRoiColor();
                latestResult = null;
                viewer.ClearLineDetectionPreview();
                viewer.ClearLineDetectionResult();
                UpdateResultViews();
                return;
            }

            LineDetectionParams parameters = CreateParams();
            RoiItem roi = GetCurrentLineRoi();
            if (roi != null)
            {
                roi.Color = System.Drawing.Color.DeepSkyBlue;
                viewer.ShowLineDetectionPreview(roi.ToLineDetectionFrame(), parameters);
            }
            else
            {
                ResetRoiColor();
                latestResult = null;
                viewer.ClearLineDetectionPreview();
                viewer.ClearLineDetectionResult();
                UpdateResultViews();
                return;
            }

            if (runDetection)
            {
                latestResult = viewer.DetectLine(roi, parameters);
                roi.Color = latestResult.Success ? System.Drawing.Color.DeepSkyBlue : System.Drawing.Color.Red;
                viewer.ShowLineDetectionResult(latestResult, parameters);
            }
            else
            {
                latestResult = null;
                viewer.ClearLineDetectionResult();
            }
            UpdateResultViews();
        }

        private void ResetRoiColor()
        {
            foreach (RoiItem roi in viewer.Rois)
            {
                roi.Color = System.Drawing.Color.DeepSkyBlue;
            }
        }

        private RoiItem GetCurrentLineRoi()
        {
            return viewer.SelectedRoi != null && viewer.SelectedRoi.CanDetectLine()
                ? viewer.SelectedRoi
                : viewer.Rois.FirstOrDefault(x => x.CanDetectLine());
        }

        private LineDetectionParams CreateParams()
        {
            return new LineDetectionParams
            {
                EdgeThreshold = ParseFloat(ThresholdTextBox.Text, 20f),
                SampleCount = ParseInt(SampleCountTextBox.Text, 40),
                EdgePolarity = GetSelectedValue(PolarityComboBox, LineEdgePolarity.Any),
                StrengthType = GetSelectedValue(StrengthTypeComboBox, LineEdgeStrengthType.Gradient1D),
                SelectionMode = GetSelectedValue(SelectionModeComboBox, LineSelectionMode.Strongest),
                FitMode = GetSelectedValue(FitModeComboBox, LineFitMode.Robust),
                ScanDirection = currentDirection
            };
        }

        private static T GetSelectedValue<T>(ComboBox comboBox, T defaultValue)
        {
            return comboBox.SelectedItem is ComboOption<T> option ? option.Value : defaultValue;
        }

        private void UpdateDirectionButtons()
        {
            LeftToRightButton.IsChecked = currentDirection == LineScanDirection.LeftToRight;
            TopToBottomButton.IsChecked = currentDirection == LineScanDirection.TopToBottom;
            BottomToTopButton.IsChecked = currentDirection == LineScanDirection.BottomToTop;
            RightToLeftButton.IsChecked = currentDirection == LineScanDirection.RightToLeft;
        }

        private void UpdateResultViews()
        {
            if (latestResult == null)
            {
                SetStatusText("未检测", false);
                LiveLineTextBlock.Text = "-";
                ResultStatusTextBlock.Text = "未检测";
                ResultStartTextBlock.Text = "起点: -";
                ResultEndTextBlock.Text = "终点: -";
                ResultAngleTextBlock.Text = "角度: -";
                ResultPointCountTextBlock.Text = "检测点数: -";
                ResultAverageStrengthTextBlock.Text = "平均边缘强度: -";
                ResultMaxStrengthTextBlock.Text = "最大边缘强度: -";
                ResultDirectionTextBlock.Text = "检测方向: -";
                return;
            }

            SetStatusText(latestResult.Message, latestResult.Success);
            LiveLineTextBlock.Text = latestResult.Success
                ? string.Format("起点 {0}，终点 {1}，角度 {2:F2}°", FormatPoint(latestResult.LineStart), FormatPoint(latestResult.LineEnd), latestResult.Angle)
                : string.Format("检测点数 {0}", latestResult.EdgePoints.Count);

            ResultStatusTextBlock.Text = latestResult.Message;
            ResultStatusTextBlock.Foreground = latestResult.Success ? Brushes.ForestGreen : Brushes.Red;
            ResultStartTextBlock.Text = "起点: " + (latestResult.Success ? FormatPoint(latestResult.LineStart) : "-");
            ResultEndTextBlock.Text = "终点: " + (latestResult.Success ? FormatPoint(latestResult.LineEnd) : "-");
            ResultAngleTextBlock.Text = "角度: " + (latestResult.Success ? latestResult.Angle.ToString("F2", CultureInfo.InvariantCulture) + "°" : "-");
            ResultPointCountTextBlock.Text = "检测点数: " + latestResult.EdgePoints.Count.ToString(CultureInfo.InvariantCulture);
            ResultAverageStrengthTextBlock.Text = "平均边缘强度: " + latestResult.AverageStrength.ToString("F2", CultureInfo.InvariantCulture);
            ResultMaxStrengthTextBlock.Text = "最大边缘强度: " + latestResult.MaxStrength.ToString("F2", CultureInfo.InvariantCulture);
            ResultDirectionTextBlock.Text = "检测方向: " + GetDirectionText(latestResult.ScanDirection);
        }

        private void SetStatusText(string text, bool success)
        {
            LiveStatusTextBlock.Text = text;
            LiveStatusTextBlock.Foreground = success ? Brushes.ForestGreen : Brushes.Red;
        }

        private static string FormatPoint(PointF point)
        {
            return string.Format(CultureInfo.InvariantCulture, "({0:F2}, {1:F2})", point.X, point.Y);
        }

        private static string GetDirectionText(LineScanDirection direction)
        {
            switch (direction)
            {
                case LineScanDirection.RightToLeft:
                    return "从右到左";
                case LineScanDirection.TopToBottom:
                    return "从上到下";
                case LineScanDirection.BottomToTop:
                    return "从下到上";
                default:
                    return "从左到右";
            }
        }

        private static float ParseFloat(string text, float defaultValue)
        {
            return float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) ? value : defaultValue;
        }

        private static int ParseInt(string text, int defaultValue)
        {
            return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) ? value : defaultValue;
        }

        private enum OperatorModule
        {
            Input,
            Roi,
            Params,
            Result
        }

        private sealed class ComboOption<T>
        {
            public ComboOption(string text, T value)
            {
                Text = text;
                Value = value;
            }

            public string Text { get; private set; }

            public T Value { get; private set; }

            public override string ToString()
            {
                return Text;
            }
        }
    }
}
