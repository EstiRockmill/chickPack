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
    public class BoxService : IBoxService
    {
        private readonly IBoxRepository _boxRepository;

        public BoxService(IBoxRepository boxRepository)
        {
            this._boxRepository = boxRepository;
        }

        public List<Box> GetBoxList()
        {
            return _boxRepository.GetBoxList();
        }

        public Box GetBoxById(int id)
        {
            return _boxRepository.GetBoxById(id);
        }

        public void DeleteBox(int id)
        {
            _boxRepository.DeleteBox(id);
        }

        public Box UpdateBox(Box box, int id)
        {
            return _boxRepository.UpdateBox(box, id);
        }

        public Box AddBox(Box box)
        {
            return _boxRepository.AddBox(box);
        }
    }
}
