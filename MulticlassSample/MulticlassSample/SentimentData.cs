using Microsoft.ML.Data;

namespace MulticlassSample
{
    public class SentimentData
    {
        [LoadColumn(0), ColumnName("SentimentText")]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Sentiment")]
        public string Sentiment;
    }

    public class SentimentPrediction
    {

        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }

        public float Probability { get; set; }
    }
}
