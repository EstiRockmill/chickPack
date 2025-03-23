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
    public class BoxRepository : IBoxRepository
    {
        private readonly DataContext _context;

        public BoxRepository(DataContext context)
        {
            _context = context;
        }
        public Box AddBox(Box box)
        {
            _context.Boxes.Add(box);
            _context.SaveChanges();
            return box;
        }

        public void DeleteBox(int id)
        {
            Box b = GetBoxById(id);
            _context.Boxes.Remove(b);
            _context.SaveChanges();
        }

        public Box GetBoxById(int id)
        {
            return _context.Boxes.FirstOrDefault(b => b.RoomId == id);
        }

        public List<Box> GetBoxList()
        {
            return _context.Boxes.Include(b=>b.Items).ToList();
        }

        public Box UpdateBox(Box box, int BoxId)
        {
            Box b = GetBoxById(BoxId);
             b.Capacity = box.Capacity;
            b.Contents = box.Contents;
            b.Items = box.Items;
            b.Room = box.Room;
            b.RoomId=box.RoomId;
            _context.Boxes.Update(b);
            _context.SaveChanges();
            return b;
        }
    }
}
