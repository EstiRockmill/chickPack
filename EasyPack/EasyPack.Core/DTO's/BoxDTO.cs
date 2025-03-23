using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.DTO_s
{
    public class BoxDTO
    {
        public string Contents { get; set; }
        public double Capacity { get; set; }
        public List<ItemDTO> Items { get; set; }
        public int RoomId { get; set; }
    }
}
