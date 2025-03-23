using EasyPack.Core.Models;
using EasyPack.Core.Repositories;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Data.Repositories
{
    public class CategoryRepository : ICategoryRepository
    {
        private readonly DataContext _context;

        public CategoryRepository(DataContext context)
        {
            _context = context;
        }
        public Category AddCategory(Category category)
        {
            _context.Categories.Add(category);
            _context.SaveChanges();
            return category;
        }

        public void DeleteCategory(int id)
        {
            Category b = GetCategoryById(id);
            _context.Categories.Remove(b);
            _context.SaveChanges();
        }

        public Category GetCategoryById(int id)
        {
            return _context.Categories.FirstOrDefault(b => b.Id == id);
        }

        public List<Category> GetCategoryList()
        {
            return _context.Categories.Include(c=>c.Items).ToList();
        }

        public Category UpdateCategory(Category category, int CategoryId)
        {
            Category c = GetCategoryById(CategoryId);
            c.Items = category.Items;
            c.Name = category.Name;
            _context.Categories.Update(c);
            _context.SaveChanges();
            return c;
        }
    }
}
