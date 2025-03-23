using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Service
{
    public interface IFirstFit
    {
        public int FirstFitPack(Node root, List<Item> items, double c);
        public void FindFirstPlace(Item item, double c, Node root);

    }
}
