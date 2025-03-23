using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Repositories
{
    public interface IItemRepository
    {
        List<Item> GetItemList();
        Item GetItemById(int id);
        void DeleteItem(int i);
        Item UpdateItem(Item item, int itemId);
        Item AddItem(Item item);
    }
}
