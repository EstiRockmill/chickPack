using EasyPack.Core.Models;
using EasyPack.Core.Repositories;
using EasyPack.Core.Service;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Service.Service
{
    public class CategoryService:ICategoryService
    {
        private readonly ICategoryRepository _categoryRepository;

        public CategoryService(ICategoryRepository categoryRepository)
        {
            this._categoryRepository = categoryRepository;
        }

        public List<Category> GetCategoryList()
        {
            return _categoryRepository.GetCategoryList();
        }

        public Category GetCategoryById(int id)
        {
            return _categoryRepository.GetCategoryById(id);
        }

        public void DeleteCategory(int id)
        {
            _categoryRepository.DeleteCategory(id);
        }

        public Category UpdateCategory(Category category, int id)
        {
            return _categoryRepository.UpdateCategory(category, id);
        }

        public Category AddCategory(Category category)
        {
            return _categoryRepository.AddCategory(category);
        }
    }
}
