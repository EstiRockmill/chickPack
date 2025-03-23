using AutoMapper;
using EasyPack.API.Models.Box;
using EasyPack.Core.DTO_s;
using EasyPack.Core.Models;
using EasyPack.Core.Service;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;


namespace EasyPack.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class BoxController : ControllerBase
    {
        private readonly IBoxService _boxService;
        private readonly IMapper _mapper;
        public BoxController(IBoxService boxService, IMapper mapper)
        {
            _boxService = boxService;
            _mapper = mapper;
        }

        [HttpGet]
        [Authorize(Roles = "user")]
        public List<BoxDTO> Get()
        {
            var boxes = _boxService.GetBoxList();
            var boxesDTO = _mapper.Map<List<BoxDTO>>(boxes);
            return boxesDTO;
        }

        // GET api/<BoxController>/5
        [HttpGet("{id}")]
        [Authorize(Roles = "user")]

        public BoxDTO GetBoxByRoom(int id)
        {
            var box = _boxService.GetBoxById(id);
            var boxDTO = _mapper.Map<BoxDTO>(box);
            return boxDTO;
        }

        // POST api/<BoxController>
        [HttpPost]
        [Authorize(Roles = "user")]

        public BoxDTO Post([FromBody] BoxPostModel box)
        {
            var b = _mapper.Map<Box>(box);
            return _mapper.Map<BoxDTO>(_boxService.AddBox(b));
        }

        [HttpPut("{id}")]
        [Authorize(Roles = "user")]

        public BoxDTO Put(int id, [FromBody] BoxPutModel box)
        {
            var b = _mapper.Map<Box>(box);

            return _mapper.Map<BoxDTO>(_boxService.UpdateBox(b, id));
        }

        [HttpDelete("{id}")]
        [Authorize(Roles = "user")]

        public void Delete(int id)
        {
            _boxService.DeleteBox(id);
        }
    }
}
