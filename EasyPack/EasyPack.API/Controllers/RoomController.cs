using AutoMapper;
using EasyPack.API.Models.Room;
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
    public class RoomController : ControllerBase
    {
        private readonly IRoomService _roomService;
        private readonly IMapper _mapper;

        public RoomController(IRoomService roomService,IMapper mapper)
        {
            _roomService = roomService;
            _mapper = mapper;
        }

        [HttpGet]
        [Authorize(Roles = "user")]

        public List<RoomDTO> Get()
        {
            return _mapper.Map<List<RoomDTO>>( _roomService.GetRoomList());
        }

        // GET api/<RoomController>/5
        [HttpGet("{id}")]
        [Authorize(Roles = "user")]


        public RoomDTO Get(int id)
        {
            return _mapper.Map<RoomDTO>( _roomService.GetRoomById(id));
        }

        // POST api/<RoomController>
        [HttpPost]
        [Authorize(Roles = "user")]

        public RoomDTO Post([FromBody] RoomPostModel room)
        {
            return _mapper.Map<RoomDTO>(_roomService.AddRoom(_mapper.Map<Room>(room)));
        }

        // PUT api/<RoomController>/5
        [HttpPut("{id}")]
        [Authorize(Roles = "user")]

        public RoomDTO Put(int id, [FromBody] RoomPutModel room)
        {
            return _mapper.Map<RoomDTO>( _roomService.UpdateRoom(_mapper.Map<Room>(room), id));
        }

        // DELETE api/<RoomController>/5
        [HttpDelete("{id}")]
        [Authorize(Roles = "user")]

        public void Delete(int id)
        {
            _roomService.DeleteRoom(id);
        }
    }
}
