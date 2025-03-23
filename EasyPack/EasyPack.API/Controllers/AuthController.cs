using AutoMapper;
using EasyPack.API.Models.User;
using EasyPack.Core.Models;
using EasyPack.Core.Service;
using Library.Core.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace LibraryAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class AuthController : ControllerBase
    {
        private readonly IUserService _userService;
        private readonly IMapper _mapper;
        private readonly IConfiguration _configuration;

        public AuthController(IConfiguration configuration, IUserService userService, IMapper mapper)
        {
            _configuration = configuration;
            _userService = userService;
            _mapper = mapper;
        }

        [HttpPost("login")]
        public IActionResult Login([FromBody] LoginModel loginModel)
        {
            var user = _userService.GetUserByEmail(loginModel.Email);
            if (user == null)
            {
                return NotFound("User not found.");
            }
            if (!BCrypt.Net.BCrypt.Verify(loginModel.Password, user.Password))
            {
                return Unauthorized("Incorrect password.");
            }

            var claims = new List<Claim>()
    {
        new Claim(ClaimTypes.Name, user.Name),
        new Claim(ClaimTypes.Role, "user")
    };

            var secretKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_configuration.GetValue<string>("JWT:Key")));
            var signinCredentials = new SigningCredentials(secretKey, SecurityAlgorithms.HmacSha256);

            var tokenOptions = new JwtSecurityToken(
                issuer: _configuration.GetValue<string>("JWT:Issuer"),
                audience: _configuration.GetValue<string>("JWT:Audience"),
                claims: claims,
                expires: DateTime.Now.AddMinutes(60),
                signingCredentials: signinCredentials
            );

            var tokenString = new JwtSecurityTokenHandler().WriteToken(tokenOptions);
            return Ok(new { Token = tokenString, User = user });
        }


        [HttpPost("signup")]
        public ActionResult<User> SignUp([FromBody] UserPostModel signUpModel)

        {
            if (_userService.GetUserByEmail(signUpModel.Email) != null)
            {
                return BadRequest("Email already in use.");
            }

            var hashedPassword = BCrypt.Net.BCrypt.HashPassword(signUpModel.Password);
            User newUser = new User
            {
                Name = signUpModel.Name,
                Email = signUpModel.Email,
                Password = hashedPassword,
                Address = signUpModel.Address,
            };
            var createdUser = _userService.AddUser(newUser);
            var claims = new List<Claim>()
                    {
                    new Claim(ClaimTypes.Name, createdUser.Name),
                    new Claim(ClaimTypes.Role, "user")
                    };
            var secretKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_configuration["JWT:Key"]));
            var signinCredentials = new SigningCredentials(secretKey, SecurityAlgorithms.HmacSha256);

            var tokenOptions = new JwtSecurityToken(
                issuer: _configuration.GetValue<string>("JWT:Issuer"),
                audience: _configuration.GetValue<string>("JWT:Audience"),
                claims: claims,
                expires: DateTime.Now.AddMinutes(60),
                signingCredentials: signinCredentials
            );

            var tokenString = new JwtSecurityTokenHandler().WriteToken(tokenOptions);

            return CreatedAtAction(nameof(SignUp), new { id = createdUser.Id }, new { User = createdUser, Token = tokenString });
        }


    }

}
