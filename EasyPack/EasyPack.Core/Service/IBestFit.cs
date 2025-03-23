using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Service
{
    public interface IBestFit
    {
        public Node getRoot(Node node);
        public int bestFitPack(Node root, List<Item> items, double c);
        public void createNewBoxWithItem(Item item, double c, Node root);
        public Node findBestFitNode(Item item, Node root);
        public void findSmallestSpace(Item item, double c, Node root);

    }
}
