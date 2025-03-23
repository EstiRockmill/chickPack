using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Models
{
    public class Item
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public double Capacity { get; set; }
        public int? BoxId { get; set; }
        [ForeignKey("BoxId")]
        public Box? Box{ get; set; }
        
    }
}
