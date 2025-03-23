using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Service
{
    public interface IAVLTreeService
    {
        public int Size(Node root);
        public void RotateRight(Node x);
        public void RotateLeft(Node x);
        public void RebalanceRight(Node n);
        public void RebalanceLeft(Node n);
        public void Rebalance(Node n);
        public Node Find(Box k, Node r);
        public Node Insert(Box k, Node root);
        public void AVLInsert(Box k, Node r);
        public Node Next(Node n);
        public Node Prev(Node n);
        public void Swap(Node a, Node b);
        public Node Delete(Node toDelete);
        public Node DeleteAVL(Node toDelete);
    }
}
