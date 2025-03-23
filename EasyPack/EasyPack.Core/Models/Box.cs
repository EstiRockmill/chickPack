using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Models
{
    public class Box
    {
        public int Id { get; set; }
        public string Contents { get; set; }
        public double Capacity { get; set; }
        public List<Item> Items { get; set; }
        public int RoomId { get; set; }
        [ForeignKey("RoomId")]
        public Room Room { get; set; }

        public Box(double capacity)
        {
            Capacity = capacity;
            Items = new List<Item>();
        }
    }
}
