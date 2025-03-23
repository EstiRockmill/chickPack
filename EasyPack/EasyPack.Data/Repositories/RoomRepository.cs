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
    public class RoomRepository:IRoomRepository
    {
        private readonly DataContext _context;

        public RoomRepository(DataContext context)
        {
            _context = context;
        }
        public Room AddRoom(Room room)
        {
            _context.Rooms.Add(room);
            _context.SaveChanges();
            return room;
        }

        public void DeleteRoom(int id)
        {
            Room b = GetRoomById(id);
            _context.Rooms.Remove(b);
            _context.SaveChanges();
        }

        public Room GetRoomById(int id)
        {
            return _context.Rooms.FirstOrDefault(b => b.Id == id);
        }

        public List<Room> GetRoomList()
        {
            return _context.Rooms.Include(r=>r.Boxes).ToList();
        }

        public Room UpdateRoom(Room room, int RoomId)
        {
            Room r = GetRoomById(RoomId);
            r.Name = room.Name;
            r.Boxes= room.Boxes;
            r.User= room.User;
            r.UserId= room.UserId;
            r.NumOfBoxes= room.NumOfBoxes;
            _context.Rooms.Update(r);
            _context.SaveChanges();
            return r;
        }
    }
}
