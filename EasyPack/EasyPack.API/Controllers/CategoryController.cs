using AutoMapper;
using EasyPack.API.Models.Category;
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
    public class CategoryController : ControllerBase
    {
        private readonly ICategoryService _categoryService;
        private readonly IMapper _mapper;

        public CategoryController(ICategoryService categoryService, IMapper mapper)
        {
            _categoryService = categoryService;
            _mapper = mapper;

        }

        [HttpGet]
        [Authorize(Roles = "user")]

        public List<CategoryDTO> Get()
        {
            return _mapper.Map<List<CategoryDTO>>(_categoryService.GetCategoryList());
        }

        // GET api/<CategoryController>/5
        [HttpGet("{id}")]
        [Authorize(Roles = "user")]

        public CategoryDTO Get(int id)
        {
            return _mapper.Map<CategoryDTO>(_categoryService.GetCategoryById(id));
        }

        // POST api/<CategoryController>
        [HttpPost]
        [Authorize(Roles = "user")]

        public CategoryDTO Post([FromBody] CategoryPostModel category)
        {
            var c = _mapper.Map<Category>(category);
            return _mapper.Map<CategoryDTO>(_categoryService.AddCategory(c));
        }

        // PUT api/<CategoryController>/5
        [HttpPut("{id}")]
        [Authorize(Roles = "user")]

        public CategoryDTO Put(int id, [FromBody] CategoryPostModel category)
        {
            return _mapper.Map<CategoryDTO>(
                _categoryService.UpdateCategory(_mapper.Map<Category>(category), id));
        }

        // DELETE api/<CategoryController>/5
        [HttpDelete("{id}")]
        [Authorize(Roles = "user")]

        public void Delete(int id)
        {
            _categoryService.DeleteCategory(id);
        }
    }
}
