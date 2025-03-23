using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Service
{
    public interface IItemService
    {
        List<Item> GetItemList();
        Item GetItemById(int id);
        void DeleteItem(int id);
        Item UpdateItem(Item item, int itemId);
        Item AddItem(Item item);
    }
}
