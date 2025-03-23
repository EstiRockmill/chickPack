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
    public class RoomService:IRoomService
    {
        private readonly IRoomRepository _roomRepository;

        public RoomService(IRoomRepository roomRepository)
        {
            this._roomRepository = roomRepository;
        }

        public List<Room> GetRoomList()
        {
            return _roomRepository.GetRoomList();
        }

        public Room GetRoomById(int id)
        {
            return _roomRepository.GetRoomById(id);
        }

        public void DeleteRoom(int id)
        {
            _roomRepository.DeleteRoom(id);
        }

        public Room UpdateRoom(Room room, int id)
        {
            return _roomRepository.UpdateRoom(room, id);
        }

        public Room AddRoom(Room room)
        {
            return _roomRepository.AddRoom(room);
        }
    }
}
