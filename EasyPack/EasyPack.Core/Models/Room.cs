using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace EasyPack.Core.Models
{
    public class Room
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public int NumOfBoxes { get; set; } = 0;
        [JsonIgnore]
        public List<Box>? Boxes { get; set; }
        public int UserId { get; set; }
        [ForeignKey("UserId")]
        public User User { get; set; }
    }
}
