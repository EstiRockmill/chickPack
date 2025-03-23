using EasyPack.Core.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;
using System.Diagnostics;
using System.Net.Http.Headers;

namespace EasyPack.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ImageAnalysisRequest
    {

        public IFormFile Image1 { get; set; }
        public IFormFile Image2 { get; set; }
    }
    public class ImageAnalysisResult
    {
        public string Label { get; set; }

        [JsonProperty("volume_cm3")]
        public float VolumeCm3 { get; set; }
    }


    public class ImageAnalysisController : ControllerBase
    {
        private readonly HttpClient _httpClient;

        public ImageAnalysisController(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }
        [HttpPost("analyze")]
        public async Task<IActionResult> AnalyzeImages([FromForm] ImageAnalysisRequest imageAnalysisRequest)
        {
            if (imageAnalysisRequest.Image1 == null || imageAnalysisRequest.Image2 == null)
            {
                return BadRequest("שתי תמונות חייבות להישלח");
            }

            using var content = new MultipartFormDataContent();

            // המרת הקבצים לבינארי ושליחתם
            using var stream1 = imageAnalysisRequest.Image1.OpenReadStream();
            using var stream2 = imageAnalysisRequest.Image2.OpenReadStream();

            var imageContent1 = new StreamContent(stream1);
            imageContent1.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");

            var imageContent2 = new StreamContent(stream2);
            imageContent2.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");

            content.Add(imageContent1, "image1", imageAnalysisRequest.Image1.FileName);
            content.Add(imageContent2, "image2", imageAnalysisRequest.Image2.FileName);

            var response = await _httpClient.PostAsync("http://localhost:5000/analyze-images", content);

            if (!response.IsSuccessStatusCode)
            {
                return StatusCode((int)response.StatusCode, "שגיאה בניתוח התמונה");
            }

            var jsonResponse = await response.Content.ReadAsStringAsync();
            var result = JsonConvert.DeserializeObject<List<ImageAnalysisResult>>(jsonResponse);

            return Ok(JsonConvert.SerializeObject(result));
        }

        ///רבים
        //[HttpPost("analyze")]
        //public async Task<IActionResult> AnalyzeImages([FromForm] ImageAnalysisRequest imageAnalysisRequest)
        //{
        //    if (imageAnalysisRequest.Images == null || !imageAnalysisRequest.Images.Any())
        //    {
        //        return BadRequest("עליך לשלוח לפחות תמונה אחת");
        //    }

        //    var results = new List<string>();  // רשימה שתשמור את כל ה-JSONים של התשובות מהפייתון

        //    foreach (var image in imageAnalysisRequest.Images)
        //    {
        //        using var content = new MultipartFormDataContent();

        //        // המרת התמונה לבינארי
        //        using var stream = image.OpenReadStream();

        //        var imageContent1 = new StreamContent(stream);
        //        imageContent1.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");

        //        // יצירת עותק נוסף של אותה התמונה
        //        var imageContent2 = new StreamContent(stream);
        //        imageContent2.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");

        //        // הוספת שתי התמונות לאותו MultipartFormDataContent עם שמות שונים
        //        content.Add(imageContent1, "image1", image.FileName); // תמונה אחת
        //        content.Add(imageContent2, "image2", image.FileName); // תמונה שנייה (אותה תמונה בפועל)

        //        // שליחה לשרת פייתון
        //        var response = await _httpClient.PostAsync("http://localhost:5000/analyze-images", content);

        //        if (!response.IsSuccessStatusCode)
        //        {
        //            return StatusCode((int)response.StatusCode, "שגיאה בניתוח התמונה");
        //        }

        //        var jsonResponse = await response.Content.ReadAsStringAsync();
        //        results.Add(jsonResponse);  // הוספת התשובה לרשימה
        //    }

        //    // החזרת התשובות כ-JSON אחד הכולל את כל התשובות של התמונות
        //    return Content($"[{string.Join(",", results)}]", "application/json");
        //}




    }
}

