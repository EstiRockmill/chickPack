using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Models
{
    public class Node
    {
        public Box Box { get; set; }
        public Node Parent { get; set; }
        public Node Left { get; set; }
        public Node Right { get; set; }
        public int Height { get; set; }
        public Node(Box b,Node parent=null)
        {
            Box =b;
            Parent = parent;
            Left = Right = null;
            Height = 1;
        }
        
    }
}
