using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace OpenCvWindowTool
{
    /// <summary>
    /// OpenCV直线检测算子，使用类似Halcon卡尺的灰度剖面测量逻辑。
    /// </summary>
    public sealed class LineDetectionOperator
    {
        public LineDetectionResult Detect(Mat image, RoiItem roi, LineDetectionParams parameters)
        {
            LineDetectionParams actualParams = NormalizeParams(parameters);
            if (image == null || image.Empty())
            {
                return LineDetectionResult.CreateFailure("当前没有可检测的图像。", default(LineDetectionFrame), actualParams.ScanDirection);
            }
            if (roi == null)
            {
                return LineDetectionResult.CreateFailure("请先选择普通矩形ROI或带角度矩形ROI。", default(LineDetectionFrame), actualParams.ScanDirection);
            }
            if (!roi.CanDetectLine())
            {
                return LineDetectionResult.CreateFailure("直线检测只支持普通矩形ROI和带角度矩形ROI。", default(LineDetectionFrame), actualParams.ScanDirection);
            }

            LineDetectionFrame frame = roi.ToLineDetectionFrame();
            using (Mat gray = ToGray(image))
            {
                List<LineEdgePoint> edgePoints = CollectEdgePoints(gray, frame, actualParams);
                if (edgePoints.Count < 2)
                {
                    return LineDetectionResult.CreateFailure("有效检测点不足，无法拟合直线。", frame, actualParams.ScanDirection, edgePoints);
                }

                PointF[] line = FitLine(frame, actualParams, edgePoints);
                return LineDetectionResult.CreateSuccess(frame, actualParams.ScanDirection, line[0], line[1], edgePoints);
            }
        }

        private static LineDetectionParams NormalizeParams(LineDetectionParams parameters)
        {
            LineDetectionParams source = parameters ?? new LineDetectionParams();
            LineDetectionParams result = new LineDetectionParams
            {
                EdgeThreshold = Math.Max(0f, source.EdgeThreshold),
                SampleCount = Math.Max(2, source.SampleCount),
                SampleStep = Math.Max(0.25f, source.SampleStep),
                SmoothSize = Math.Max(1, source.SmoothSize),
                EdgePolarity = source.EdgePolarity,
                StrengthType = source.StrengthType,
                SelectionMode = source.SelectionMode,
                ScanDirection = source.ScanDirection,
                FitMode = source.FitMode
            };
            if (result.SmoothSize % 2 == 0) result.SmoothSize++;
            return result;
        }

        private static Mat ToGray(Mat image)
        {
            Mat gray = new Mat();
            if (image.Channels() == 1)
            {
                image.ConvertTo(gray, MatType.CV_8U);
            }
            else if (image.Channels() == 3)
            {
                Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
            }
            else if (image.Channels() == 4)
            {
                Cv2.CvtColor(image, gray, ColorConversionCodes.BGRA2GRAY);
            }
            else
            {
                image.ConvertTo(gray, MatType.CV_8U);
            }
            return gray;
        }

        private static List<LineEdgePoint> CollectEdgePoints(Mat gray, LineDetectionFrame frame, LineDetectionParams parameters)
        {
            PointF scanDir = frame.GetScanDirection(parameters.ScanDirection);
            PointF arrangeDir = frame.GetLineArrangeDirection(parameters.ScanDirection);
            float arrangeLength = frame.GetArrangeLength(parameters.ScanDirection);
            float scanLength = frame.GetScanLength(parameters.ScanDirection);
            float caliperHalfWidth = Math.Max(0.5f, arrangeLength / Math.Max(1, parameters.SampleCount) * 0.5f);
            float arrangeStart = -arrangeLength / 2f + caliperHalfWidth;
            float arrangeEnd = arrangeLength / 2f - caliperHalfWidth;
            float arrangeStep = parameters.SampleCount == 1 ? 0f : (arrangeEnd - arrangeStart) / (parameters.SampleCount - 1);
            List<List<CaliperCandidate>> candidateGroups = new List<List<CaliperCandidate>>();

            using (Mat sobelX = new Mat())
            using (Mat sobelY = new Mat())
            {
                if (parameters.StrengthType == LineEdgeStrengthType.Sobel)
                {
                    Cv2.Sobel(gray, sobelX, MatType.CV_32F, 1, 0, 3);
                    Cv2.Sobel(gray, sobelY, MatType.CV_32F, 0, 1, 3);
                }

                for (int i = 0; i < parameters.SampleCount; i++)
                {
                    float along = arrangeStart + arrangeStep * i;
                    PointF center = new PointF(frame.Center.X + arrangeDir.X * along, frame.Center.Y + arrangeDir.Y * along);
                    List<CaliperCandidate> candidates = DetectOneCaliper(gray, sobelX, sobelY, center, scanDir, arrangeDir, scanLength, caliperHalfWidth, i, parameters);
                    candidateGroups.Add(candidates);
                }
            }

            return SelectGloballyConsistentPoints(candidateGroups, frame, parameters);
        }

        private static List<CaliperCandidate> DetectOneCaliper(Mat gray, Mat sobelX, Mat sobelY, PointF center, PointF scanDir, PointF widthDir, float scanLength, float halfWidth, int caliperIndex, LineDetectionParams parameters)
        {
            List<ProfileSample> profile = BuildAveragedProfile(gray, sobelX, sobelY, center, scanDir, widthDir, scanLength, halfWidth, parameters);
            if (profile.Count < 3) return new List<CaliperCandidate>();

            Smooth(profile, parameters.SmoothSize);
            return FindCandidates(profile, caliperIndex, parameters);
        }

        private static List<ProfileSample> BuildAveragedProfile(Mat gray, Mat sobelX, Mat sobelY, PointF center, PointF scanDir, PointF widthDir, float scanLength, float halfWidth, LineDetectionParams parameters)
        {
            int scanCount = Math.Max(3, (int)Math.Floor(scanLength / parameters.SampleStep) + 1);
            int widthCount = Math.Max(1, (int)Math.Floor((halfWidth * 2f) / parameters.SampleStep) + 1);
            float scanStart = -scanLength / 2f;
            float widthStart = -halfWidth;
            List<ProfileSample> profile = new List<ProfileSample>(scanCount);

            for (int scanIndex = 0; scanIndex < scanCount; scanIndex++)
            {
                float scanOffset = scanStart + scanIndex * parameters.SampleStep;
                PointF scanPoint = new PointF(center.X + scanDir.X * scanOffset, center.Y + scanDir.Y * scanOffset);
                double graySum = 0d;
                double sobelSum = 0d;
                int validCount = 0;

                for (int widthIndex = 0; widthIndex < widthCount; widthIndex++)
                {
                    float widthOffset = widthStart + widthIndex * parameters.SampleStep;
                    PointF p = new PointF(scanPoint.X + widthDir.X * widthOffset, scanPoint.Y + widthDir.Y * widthOffset);
                    if (!IsInside(gray, p)) continue;

                    graySum += ReadGrayBilinear(gray, p);
                    if (parameters.StrengthType == LineEdgeStrengthType.Sobel)
                    {
                        sobelSum += ReadDirectionalSobel(sobelX, sobelY, p, scanDir);
                    }
                    validCount++;
                }

                if (validCount == 0) continue;
                float grayValue = (float)(graySum / validCount);
                float sobelValue = parameters.StrengthType == LineEdgeStrengthType.Sobel ? (float)(sobelSum / validCount) : 0f;
                profile.Add(new ProfileSample(scanPoint, scanOffset, grayValue, sobelValue));
            }

            return profile;
        }

        private static List<CaliperCandidate> FindCandidates(List<ProfileSample> profile, int caliperIndex, LineDetectionParams parameters)
        {
            List<CaliperCandidate> candidates = new List<CaliperCandidate>();
            for (int i = 1; i < profile.Count - 1; i++)
            {
                float gradient = CalculateGradient(profile, i, parameters.StrengthType);
                float strength = Math.Abs(gradient);
                if (strength < parameters.EdgeThreshold) continue;
                if (!MatchPolarity(gradient, parameters.EdgePolarity)) continue;
                if (!IsLocalPeak(profile, i, parameters.StrengthType)) continue;

                PointF point = RefineEdgePoint(profile, i, parameters.StrengthType);
                candidates.Add(new CaliperCandidate(point, profile[i].Offset, strength, caliperIndex));
            }

            return candidates;
        }

        private static List<LineEdgePoint> SelectGloballyConsistentPoints(List<List<CaliperCandidate>> candidateGroups, LineDetectionFrame frame, LineDetectionParams parameters)
        {
            List<CaliperCandidate> allCandidates = candidateGroups.SelectMany(x => x).ToList();
            if (allCandidates.Count < 2)
            {
                return SelectFallbackPoints(candidateGroups, parameters);
            }

            PointF arrangeDir = frame.GetLineArrangeDirection(parameters.ScanDirection);
            float tolerance = Math.Max(2.0f, Math.Min(8.0f, frame.GetScanLength(parameters.ScanDirection) * 0.04f));
            List<CaliperCandidate> hypothesisCandidates = candidateGroups
                .SelectMany(x => TakeBestCandidates(x, parameters, 4))
                .ToList();
            LineHypothesis best = FindBestLineHypothesis(hypothesisCandidates, candidateGroups, arrangeDir, tolerance, parameters);
            if (!best.IsValid)
            {
                return SelectFallbackPoints(candidateGroups, parameters);
            }

            List<LineEdgePoint> selected = SelectInliers(candidateGroups, best, tolerance * 1.75f, parameters);
            if (selected.Count < 2)
            {
                return SelectFallbackPoints(candidateGroups, parameters);
            }

            return selected;
        }

        private static IEnumerable<CaliperCandidate> TakeBestCandidates(List<CaliperCandidate> candidates, LineDetectionParams parameters, int maxCount)
        {
            if (candidates == null || candidates.Count == 0) return Enumerable.Empty<CaliperCandidate>();
            return candidates
                .OrderByDescending(x => CandidatePreference(x, candidates, parameters))
                .Take(maxCount);
        }

        private static LineHypothesis FindBestLineHypothesis(List<CaliperCandidate> candidates, List<List<CaliperCandidate>> groups, PointF arrangeDir, float tolerance, LineDetectionParams parameters)
        {
            LineHypothesis best = LineHypothesis.Invalid;
            for (int i = 0; i < candidates.Count - 1; i++)
            {
                for (int j = i + 1; j < candidates.Count; j++)
                {
                    if (candidates[i].CaliperIndex == candidates[j].CaliperIndex) continue;

                    LineHypothesis hypothesis = LineHypothesis.FromPoints(candidates[i].Point, candidates[j].Point);
                    if (!hypothesis.IsValid) continue;

                    float orientation = Math.Abs(Dot(hypothesis.Direction, arrangeDir));
                    if (orientation < 0.5f) continue;

                    float score = ScoreHypothesis(hypothesis, groups, tolerance, parameters);
                    score += orientation;
                    if (score > best.Score)
                    {
                        best = hypothesis.WithScore(score);
                    }
                }
            }

            return best;
        }

        private static float ScoreHypothesis(LineHypothesis hypothesis, List<List<CaliperCandidate>> groups, float tolerance, LineDetectionParams parameters)
        {
            float score = 0f;
            int continuous = 0;
            int bestContinuous = 0;

            foreach (List<CaliperCandidate> group in groups)
            {
                CaliperCandidate bestCandidate = FindBestCandidateNearLine(group, hypothesis, tolerance, parameters);
                if (bestCandidate.IsValid)
                {
                    float distance = DistanceToLine(bestCandidate.Point, hypothesis);
                    score += 1.0f + CandidatePreference(bestCandidate, group, parameters) * 0.35f + (1.0f - distance / tolerance) * 0.5f;
                    continuous++;
                    if (continuous > bestContinuous) bestContinuous = continuous;
                }
                else
                {
                    continuous = 0;
                }
            }

            score += bestContinuous * 0.2f;
            return score;
        }

        private static List<LineEdgePoint> SelectInliers(List<List<CaliperCandidate>> groups, LineHypothesis hypothesis, float tolerance, LineDetectionParams parameters)
        {
            List<LineEdgePoint> selected = new List<LineEdgePoint>();
            foreach (List<CaliperCandidate> group in groups)
            {
                CaliperCandidate candidate = FindBestCandidateNearLine(group, hypothesis, tolerance, parameters);
                if (candidate.IsValid)
                {
                    selected.Add(new LineEdgePoint(candidate.Point, candidate.Strength));
                }
            }
            return selected;
        }

        private static CaliperCandidate FindBestCandidateNearLine(List<CaliperCandidate> group, LineHypothesis hypothesis, float tolerance, LineDetectionParams parameters)
        {
            if (group == null || group.Count == 0) return CaliperCandidate.Invalid;

            CaliperCandidate best = CaliperCandidate.Invalid;
            float bestScore = float.NegativeInfinity;
            foreach (CaliperCandidate candidate in group)
            {
                float distance = DistanceToLine(candidate.Point, hypothesis);
                if (distance > tolerance) continue;

                float score = CandidatePreference(candidate, group, parameters) - distance / tolerance;
                if (score > bestScore)
                {
                    bestScore = score;
                    best = candidate;
                }
            }
            return best;
        }

        private static List<LineEdgePoint> SelectFallbackPoints(List<List<CaliperCandidate>> candidateGroups, LineDetectionParams parameters)
        {
            List<LineEdgePoint> result = new List<LineEdgePoint>();
            foreach (List<CaliperCandidate> group in candidateGroups)
            {
                if (group == null || group.Count == 0) continue;
                CaliperCandidate selected;
                switch (parameters.SelectionMode)
                {
                    case LineSelectionMode.First:
                        selected = group.OrderBy(x => x.Offset).First();
                        break;
                    case LineSelectionMode.Last:
                        selected = group.OrderByDescending(x => x.Offset).First();
                        break;
                    default:
                        selected = group.OrderByDescending(x => x.Strength).First();
                        break;
                }
                result.Add(new LineEdgePoint(selected.Point, selected.Strength));
            }
            return result;
        }

        private static float CandidatePreference(CaliperCandidate candidate, List<CaliperCandidate> group, LineDetectionParams parameters)
        {
            float maxStrength = Math.Max(0.0001f, group.Max(x => x.Strength));
            float strengthScore = candidate.Strength / maxStrength;
            if (parameters.SelectionMode == LineSelectionMode.Strongest)
            {
                return strengthScore;
            }

            float minOffset = group.Min(x => x.Offset);
            float maxOffset = group.Max(x => x.Offset);
            float range = Math.Max(0.0001f, maxOffset - minOffset);
            float positionScore = parameters.SelectionMode == LineSelectionMode.First
                ? 1.0f - (candidate.Offset - minOffset) / range
                : (candidate.Offset - minOffset) / range;
            return positionScore * 0.75f + strengthScore * 0.25f;
        }

        private static float CalculateGradient(List<ProfileSample> profile, int index, LineEdgeStrengthType strengthType)
        {
            if (strengthType == LineEdgeStrengthType.Sobel)
            {
                return profile[index].SobelValue;
            }

            float span = Math.Max(0.0001f, profile[index + 1].Offset - profile[index - 1].Offset);
            return (profile[index + 1].GrayValue - profile[index - 1].GrayValue) / span;
        }

        private static bool IsLocalPeak(List<ProfileSample> profile, int index, LineEdgeStrengthType strengthType)
        {
            float current = Math.Abs(CalculateGradient(profile, index, strengthType));
            float previous = index > 1 ? Math.Abs(CalculateGradient(profile, index - 1, strengthType)) : 0f;
            float next = index < profile.Count - 2 ? Math.Abs(CalculateGradient(profile, index + 1, strengthType)) : 0f;
            return current >= previous && current >= next;
        }

        private static PointF RefineEdgePoint(List<ProfileSample> profile, int index, LineEdgeStrengthType strengthType)
        {
            if (index <= 0 || index >= profile.Count - 1) return profile[index].Point;

            float left = Math.Abs(CalculateGradient(profile, index - 1, strengthType));
            float center = Math.Abs(CalculateGradient(profile, index, strengthType));
            float right = Math.Abs(CalculateGradient(profile, index + 1, strengthType));
            float denominator = left - 2f * center + right;
            if (Math.Abs(denominator) <= 0.0001f) return profile[index].Point;

            float delta = 0.5f * (left - right) / denominator;
            delta = Math.Max(-1f, Math.Min(1f, delta));
            PointF prev = profile[index - 1].Point;
            PointF next = profile[index + 1].Point;
            return new PointF(
                profile[index].Point.X + (next.X - prev.X) * 0.5f * delta,
                profile[index].Point.Y + (next.Y - prev.Y) * 0.5f * delta);
        }

        private static bool MatchPolarity(float gradient, LineEdgePolarity polarity)
        {
            switch (polarity)
            {
                case LineEdgePolarity.Positive:
                    return gradient > 0f;
                case LineEdgePolarity.Negative:
                    return gradient < 0f;
                default:
                    return true;
            }
        }

        private static PointF[] FitLine(LineDetectionFrame frame, LineDetectionParams parameters, List<LineEdgePoint> edgePoints)
        {
            Point2f[] points = edgePoints.Select(x => new Point2f(x.Point.X, x.Point.Y)).ToArray();
            Line2D line = parameters.FitMode == LineFitMode.LeastSquares
                ? Cv2.FitLine(points, DistanceTypes.L2, 0, 0.01, 0.01)
                : Cv2.FitLine(points, DistanceTypes.Welsch, 0, 0.01, 0.01);

            PointF direction = new PointF((float)line.Vx, (float)line.Vy);
            PointF point = new PointF((float)line.X1, (float)line.Y1);
            PointF arrangeDir = frame.GetLineArrangeDirection(parameters.ScanDirection);
            if (direction.X * arrangeDir.X + direction.Y * arrangeDir.Y < 0f)
            {
                direction = new PointF(-direction.X, -direction.Y);
            }

            float halfLength = frame.GetArrangeLength(parameters.ScanDirection) / 2f;
            PointF start = new PointF(point.X - direction.X * halfLength, point.Y - direction.Y * halfLength);
            PointF end = new PointF(point.X + direction.X * halfLength, point.Y + direction.Y * halfLength);
            return ClipLineToFrame(point, direction, frame, out PointF clippedStart, out PointF clippedEnd)
                ? new[] { clippedStart, clippedEnd }
                : new[] { start, end };
        }

        private static bool ClipLineToFrame(PointF point, PointF direction, LineDetectionFrame frame, out PointF start, out PointF end)
        {
            start = PointF.Empty;
            end = PointF.Empty;
            PointF[] corners = frame.GetCorners();
            List<PointF> intersections = new List<PointF>();
            for (int i = 0; i < corners.Length; i++)
            {
                PointF a = corners[i];
                PointF b = corners[(i + 1) % corners.Length];
                if (TryIntersectInfiniteLineWithSegment(point, direction, a, b, out PointF intersection))
                {
                    AddUniquePoint(intersections, intersection);
                }
            }

            if (intersections.Count < 2) return false;

            float maxDistance = float.NegativeInfinity;
            for (int i = 0; i < intersections.Count - 1; i++)
            {
                for (int j = i + 1; j < intersections.Count; j++)
                {
                    float distance = SquaredDistance(intersections[i], intersections[j]);
                    if (distance > maxDistance)
                    {
                        maxDistance = distance;
                        start = intersections[i];
                        end = intersections[j];
                    }
                }
            }
            return maxDistance > 0.0001f;
        }

        private static bool TryIntersectInfiniteLineWithSegment(PointF linePoint, PointF lineDirection, PointF segStart, PointF segEnd, out PointF intersection)
        {
            intersection = PointF.Empty;
            PointF segmentDirection = new PointF(segEnd.X - segStart.X, segEnd.Y - segStart.Y);
            float denominator = Cross(lineDirection, segmentDirection);
            if (Math.Abs(denominator) < 0.0001f) return false;

            PointF delta = new PointF(segStart.X - linePoint.X, segStart.Y - linePoint.Y);
            float u = Cross(delta, lineDirection) / denominator;
            if (u < -0.0001f || u > 1.0001f) return false;

            intersection = new PointF(segStart.X + segmentDirection.X * u, segStart.Y + segmentDirection.Y * u);
            return true;
        }

        private static void AddUniquePoint(List<PointF> points, PointF point)
        {
            foreach (PointF existing in points)
            {
                if (SquaredDistance(existing, point) < 0.01f) return;
            }
            points.Add(point);
        }

        private static float Cross(PointF a, PointF b)
        {
            return a.X * b.Y - a.Y * b.X;
        }

        private static float SquaredDistance(PointF a, PointF b)
        {
            float dx = a.X - b.X;
            float dy = a.Y - b.Y;
            return dx * dx + dy * dy;
        }

        private static float DistanceToLine(PointF point, LineHypothesis line)
        {
            float dx = point.X - line.Point.X;
            float dy = point.Y - line.Point.Y;
            return Math.Abs(dx * line.Normal.X + dy * line.Normal.Y);
        }

        private static float Dot(PointF a, PointF b)
        {
            return a.X * b.X + a.Y * b.Y;
        }

        private static bool IsInside(Mat mat, PointF point)
        {
            return point.X >= 0f && point.Y >= 0f && point.X < mat.Width - 1 && point.Y < mat.Height - 1;
        }

        private static float ReadGrayBilinear(Mat gray, PointF point)
        {
            int x0 = Math.Max(0, Math.Min(gray.Width - 2, (int)Math.Floor(point.X)));
            int y0 = Math.Max(0, Math.Min(gray.Height - 2, (int)Math.Floor(point.Y)));
            float dx = point.X - x0;
            float dy = point.Y - y0;
            float v00 = gray.Get<byte>(y0, x0);
            float v10 = gray.Get<byte>(y0, x0 + 1);
            float v01 = gray.Get<byte>(y0 + 1, x0);
            float v11 = gray.Get<byte>(y0 + 1, x0 + 1);
            float top = v00 + (v10 - v00) * dx;
            float bottom = v01 + (v11 - v01) * dx;
            return top + (bottom - top) * dy;
        }

        private static float ReadDirectionalSobel(Mat sobelX, Mat sobelY, PointF point, PointF scanDir)
        {
            if (sobelX.Empty() || sobelY.Empty()) return 0f;
            float gx = ReadFloatBilinear(sobelX, point);
            float gy = ReadFloatBilinear(sobelY, point);
            return gx * scanDir.X + gy * scanDir.Y;
        }

        private static float ReadFloatBilinear(Mat mat, PointF point)
        {
            int x0 = Math.Max(0, Math.Min(mat.Width - 2, (int)Math.Floor(point.X)));
            int y0 = Math.Max(0, Math.Min(mat.Height - 2, (int)Math.Floor(point.Y)));
            float dx = point.X - x0;
            float dy = point.Y - y0;
            float v00 = mat.Get<float>(y0, x0);
            float v10 = mat.Get<float>(y0, x0 + 1);
            float v01 = mat.Get<float>(y0 + 1, x0);
            float v11 = mat.Get<float>(y0 + 1, x0 + 1);
            float top = v00 + (v10 - v00) * dx;
            float bottom = v01 + (v11 - v01) * dx;
            return top + (bottom - top) * dy;
        }

        private static void Smooth(List<ProfileSample> profile, int size)
        {
            if (size <= 1 || profile.Count < size) return;

            int half = size / 2;
            float[] grayValues = profile.Select(x => x.GrayValue).ToArray();
            float[] sobelValues = profile.Select(x => x.SobelValue).ToArray();
            for (int i = 0; i < profile.Count; i++)
            {
                float graySum = 0f;
                float sobelSum = 0f;
                int count = 0;
                for (int j = i - half; j <= i + half; j++)
                {
                    if (j < 0 || j >= profile.Count) continue;
                    graySum += grayValues[j];
                    sobelSum += sobelValues[j];
                    count++;
                }
                profile[i] = new ProfileSample(profile[i].Point, profile[i].Offset, graySum / count, sobelSum / count);
            }
        }

        private struct ProfileSample
        {
            public readonly PointF Point;
            public readonly float Offset;
            public readonly float GrayValue;
            public readonly float SobelValue;

            public ProfileSample(PointF point, float offset, float grayValue, float sobelValue)
            {
                Point = point;
                Offset = offset;
                GrayValue = grayValue;
                SobelValue = sobelValue;
            }
        }

        private struct CaliperCandidate
        {
            public static readonly CaliperCandidate Invalid = new CaliperCandidate(PointF.Empty, 0f, 0f, -1);

            public readonly PointF Point;
            public readonly float Offset;
            public readonly float Strength;
            public readonly int CaliperIndex;

            public CaliperCandidate(PointF point, float offset, float strength, int caliperIndex)
            {
                Point = point;
                Offset = offset;
                Strength = strength;
                CaliperIndex = caliperIndex;
            }

            public bool IsValid => CaliperIndex >= 0;
        }

        private struct LineHypothesis
        {
            public static readonly LineHypothesis Invalid = new LineHypothesis(PointF.Empty, PointF.Empty, PointF.Empty, float.NegativeInfinity, false);

            public readonly PointF Point;
            public readonly PointF Direction;
            public readonly PointF Normal;
            public readonly float Score;
            public readonly bool IsValid;

            private LineHypothesis(PointF point, PointF direction, PointF normal, float score, bool isValid)
            {
                Point = point;
                Direction = direction;
                Normal = normal;
                Score = score;
                IsValid = isValid;
            }

            public static LineHypothesis FromPoints(PointF a, PointF b)
            {
                float dx = b.X - a.X;
                float dy = b.Y - a.Y;
                float length = (float)Math.Sqrt(dx * dx + dy * dy);
                if (length <= 0.0001f) return Invalid;

                PointF direction = new PointF(dx / length, dy / length);
                PointF normal = new PointF(-direction.Y, direction.X);
                return new LineHypothesis(a, direction, normal, 0f, true);
            }

            public LineHypothesis WithScore(float score)
            {
                return new LineHypothesis(Point, Direction, Normal, score, IsValid);
            }
        }
    }
}
