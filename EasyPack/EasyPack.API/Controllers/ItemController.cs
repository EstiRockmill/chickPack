using AutoMapper;
using EasyPack.API.Models.Item;
using EasyPack.Core.DTO_s;
using EasyPack.Core.Models;
using EasyPack.Core.Service;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace EasyPack.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ItemController : ControllerBase
    {
        private readonly IItemService _itemService;
        private readonly IMapper _mapper;

        public ItemController(IItemService itemService, IMapper mapper)
        {
            _itemService = itemService;
            _mapper = mapper;
        }

        [HttpGet]
        [Authorize(Roles = "user")]

        public List<ItemDTO> Get()
        {
            return _mapper.Map<List<ItemDTO>>(_itemService.GetItemList());
        }

        // GET api/<ItemController>/5
        [HttpGet("{id}")]
        [Authorize(Roles = "user")]

        public ItemDTO Get(int id)
        {
            return _mapper.Map<ItemDTO>(_itemService.GetItemById(id));
        }

        // POST api/<ItemController>
        [HttpPost]
        [Authorize(Roles = "user")]

        public ItemDTO Post([FromBody] ItemPostModel item)
        {
            return _mapper.Map<ItemDTO>(_itemService.AddItem(_mapper.Map<Item>(item)));
        }

        // PUT api/<ItemController>/5
        [HttpPut("{id}")]
        [Authorize(Roles = "user")]

        public ItemDTO Put(int id, [FromBody] ItemPutModel item)
        {
            return _mapper.Map<ItemDTO>(_itemService.UpdateItem(_mapper.Map<Item>(item), id));
        }

        // DELETE api/<ItemController>/5
        [HttpDelete("{id}")]
        [Authorize(Roles = "user")]

        public void Delete(int id)
        {
            _itemService.DeleteItem(id);
        }
    }
}
