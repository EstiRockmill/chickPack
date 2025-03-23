using AutoMapper;
using EasyPack.API.Models.User;
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
    public class UserController : ControllerBase
    {
        private readonly IUserService _userService;
        private readonly IMapper _mapper;

        public UserController(IUserService userService,IMapper mapper)
        {
            _userService = userService;
            _mapper=mapper;
        }

        [HttpGet]

        public List<UserDTO> Get()
        {
            return _mapper.Map<List<UserDTO>>( _userService.GetUserList());
        }

        // GET api/<UserController>/5
        [HttpGet("{id}")]
        public UserDTO Get(int id)
        {
            return _mapper.Map<UserDTO> (_userService.GetUserById(id));
        }

        // POST api/<UserController>
        [HttpPost]
        public UserDTO Post([FromBody] UserPostModel user)
        {
            return _mapper.Map<UserDTO>( _userService.AddUser(_mapper.Map<User>(user)));
        }

        // PUT api/<UserController>/5
        [HttpPut("{id}")]
        public UserDTO Put(int id, [FromBody] UserPostModel user)
        {
            return _mapper.Map<UserDTO>( _userService.UpdateUser(_mapper.Map<User>(user), id));
        }

        // DELETE api/<UserController>/5
        [HttpDelete("{id}")]
        public void Delete(int id)
        {
            _userService.DeleteUser(id);
        }
    }
}
