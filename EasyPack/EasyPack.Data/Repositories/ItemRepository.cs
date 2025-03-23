using EasyPack.Core.Models;
using EasyPack.Core.Repositories;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Data.Repositories
{
    public class ItemRepository:IItemRepository
    {
        private readonly DataContext _context;

        public ItemRepository(DataContext context)
        {
            _context = context;
        }
        public Item AddItem(Item item)
        {
            _context.Items.Add(item);
            _context.SaveChanges();
            return item;
        }

        public void DeleteItem(int id)
        {
            Item b = GetItemById(id);
            _context.Items.Remove(b);
            _context.SaveChanges();
        }

        public Item GetItemById(int id)
        {
            return _context.Items.FirstOrDefault(b => b.Id == id);
        }

        public List<Item> GetItemList()
        {
            return _context.Items.ToList();
        }

        public Item UpdateItem(Item item, int ItemId)
        {
            Item i = GetItemById(ItemId);
            i.Name = item.Name;
            i.Capacity = item.Capacity;
            i.Box = item.Box;
            i.BoxId = item.BoxId;
            _context.Items.Update(i);
            _context.SaveChanges();
            return i;
        }
    }
}
