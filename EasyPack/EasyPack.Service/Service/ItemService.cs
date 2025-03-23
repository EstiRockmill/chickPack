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
    public class ItemService : IItemService
    {
        private readonly IItemRepository _itemRepository;

        public ItemService(IItemRepository itemRepository)
        {
            this._itemRepository = itemRepository;
        }

        public List<Item> GetItemList()
        {
            return _itemRepository.GetItemList();
        }

        public Item GetItemById(int id)
        {
            return _itemRepository.GetItemById(id);
        }

        public void DeleteItem(int id)
        {
            _itemRepository.DeleteItem(id);
        }

        public Item UpdateItem(Item item, int id)
        {
            return _itemRepository.UpdateItem(item, id);
        }

        public Item AddItem(Item item)
        {
            return _itemRepository.AddItem(item);
        }
    }
}
