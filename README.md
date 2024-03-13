# LLM-auto-labeler
Utilize LLM's to create datasets that will power smaller models! 

LLMs are very impressive, but the bigger they are, the slower they become.

I wanted to be able to display a job advert description and highlight the different kinds of requirements. I tested different small-sized LLMs and NER, but they weren't up to the job.

Tested using the text classifier [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base), which was both fast and decently accurate. The next challenge was to fine-tune it; unfortunately, there isn't any dataset that contains, e.g., education or qualifications requirements.

I used Mistral 7B to extract the labels below from 4k job adverts in the [LinkedIn Job Postings - 2023](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings). 
You can find out how I did this in the notebook: LLM_labels_extractor.ipynb
```
Education
Certification
Qualifications
Work experience
Hard skills
Soft skills 
Benefits
Breadtext
Company culture 
Job duties
```
 
I instructed it to output in JSON format, but since the LLMs are not 100% accurate, it didn't always output only or correct JSON. I had to preprocess and clean the data beforehand. Then, I created a dataset of the extracted data along with the negatives, which I used the text in the job description that was not used or identified as a label by the LLM. 
You can find out how I did this in: DatasetCreator.ipynb

I fine-tuned the Reranker using the notebook Reranker_training.ipynb
I formated the query trigger so they had "Example of" before the label to provide the model with the intent of the query.

You can find the model on huggingface: https://huggingface.co/MarcusLoren/Reranker-job-description
I provided code for both usage for .NET (C#) in python.


Fine-tuned querys:
```
Example of education
Example of certification
Example of qualifications
Example of work experience
Example of hard skills
Example of soft skills 
Example of benefits
Example of breadtext
Example of company culture
Example of job duties
```

Usage for .NET (C#) requires Microsoft.ML to load the model and for the tokenizer: BlingFire

 ```
    public class RankerInput
    {  
        public long[] input_ids { get; set; } 
        public long[] attention_mask { get; set; }
    }
        public class RankedOutput
    {
        public float[] logits { get; set; }
    }
  _mlContext = new MLContext();
  
  var onnxModelPath = "Reranker.onnx";
  var dataView = _mlContext.Data.LoadFromEnumerable(new List<RankerInput>());
  var pipeline = _mlContext.Transforms.ApplyOnnxModel(
      modelFile: onnxModelPath,
      gpuDeviceId: 0,
      outputColumnNames: new[] { nameof(RankedOutput.logits) },
      inputColumnNames: new[] { nameof(RankerInput.input_ids), nameof(RankerInput.attention_mask) });
  rankerModel = pipeline.Fit(dataView);
  var predictionEngine = _mlContext.Model.CreatePredictionEngine<RankerInput, RankedOutput>(rankerModel);

  var tokened_input = Tokenize(["Example of education", "Requires bachelor degree"])
  
  var pred = predictionEngine.Predict(tokened_input)
  var score = pred.logits[0];  // e.g 0.99
   
   private RankerInput Tokenize(string[] pair)
   {   
         List<long> input_ids =
         [
                 0,
                 .. TokenizeText(pair[0]),
                 2,
                 2,
                 .. TokenizeText(pair[1]),
                 2,
             ];

         var attention_mask = Enumerable.Repeat((long)1, input_ids.Count).ToArray();
         return new RankerInput() { input_ids = input_ids.ToArray(), attention_mask = attention_mask }; 
 }
  
  var TokenizerModel = BlingFireUtils.LoadModel(@".\xlm_roberta_base.bin");
  public int[] TokenizeText(string text)
  {
      List<int> tokenized = new();
      foreach (var chunk in text.Split(' ').Chunk(80)) {
  
          int[] labelIds = new int[128]; 
          byte[] inBytes = Encoding.UTF8.GetBytes(string.Join(" ", chunk));
          var outputCount = BlingFireUtils2.TextToIds(TokenizerModel, inBytes, inBytes.Length, labelIds, labelIds.Length, 0);
          Array.Resize(ref labelIds, outputCount);
          tokenized.AddRange(labelIds);
      }
      return tokenized.ToArray();
  }
```