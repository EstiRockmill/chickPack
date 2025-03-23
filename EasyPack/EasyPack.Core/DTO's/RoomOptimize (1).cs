using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace EasyPack.Core.DTO_s
{
    public class RoomOptimize
    {
        public string Name { get; set; }
        public int NumOfBoxes { get; set; }
        public List<BoxDTO> Boxes { get; set; }
        public int UserId { get; set; }
    }
}