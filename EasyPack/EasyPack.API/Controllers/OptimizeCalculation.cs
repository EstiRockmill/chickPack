using AutoMapper;
using EasyPack.API.Models.Room;
using EasyPack.Core.DTO_s;
using EasyPack.Core.Models;
using EasyPack.Core.Service;
using EasyPack.Service.Service.OptimalPack;
using Microsoft.AspNetCore.Mvc;
using System.Text.Json;
using System.Text.Json.Serialization;

public class OptimizeRequest
{
    public double Capacity { get; set; }    
    //public RoomPutModel Room { get; set; }  
    public int RoomId {  get; set; }

    public List<ItemJson> itemJsons { get; set; }
}
public class ItemJson
{
    public string Label { get; set; }

    [JsonPropertyName("volume_cm3")]
    public float VolumeCm3 { get; set; }
}
namespace EasyPack.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class OptimizeCalculation : ControllerBase
    {
        private readonly IItemService _itemService;
        private readonly INodeService _nodeService;
        private readonly IMapper _mapper;
        private readonly IRoomService _roomService;
        public OptimizeCalculation(IItemService itemService, INodeService nodeService,IMapper mapper,IRoomService roomService)
        {
            _nodeService = nodeService;
            _itemService = itemService;
            _mapper = mapper;
            _roomService = roomService;
        }
        [HttpPost]
        public async Task<IActionResult> Calculation([FromBody] OptimizeRequest optimizeRequest)
        {
          
            List<Item> items = new List<Item>();
            try
            {

                foreach (var itemJson in optimizeRequest.itemJsons)
                {
                    var item = new Item
                    {
                        Name = itemJson.Label,
                        Capacity = itemJson.VolumeCm3,
                    };

                    items.Add(_itemService.AddItem(item));
                }
            }
            catch (Exception ex)
            {
                return BadRequest($"שגיאה בהמרת JSON: {ex.Message}");
            }
            Room r=_roomService.GetRoomById(optimizeRequest.RoomId); 
             Room result = _nodeService.compareBetweenTwoWays(items, optimizeRequest.Capacity,r);

            return Ok(_mapper.Map<RoomDTO>(result));
        }
    }
}
