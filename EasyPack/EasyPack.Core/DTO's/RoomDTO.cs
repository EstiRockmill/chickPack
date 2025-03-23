using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.DTO_s
{
    public class RoomDTO
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public int NumOfBoxes { get; set; } = 0;
        public List<BoxDTO> Boxes { get; set; }
        public int UserId { get; set; }
    }
}
