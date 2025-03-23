using EasyPack.Core.Models;
using EasyPack.Core.Service;
using System;

namespace EasyPack.Service.Service.OptimalPack
{
    public class AVLTreeService : IAVLTreeService
    {
        private Node root;

        public int GetHeight(Node node) => node == null ? 0 : node.Height;
        public int GetBalance(Node node) => node == null ? 0 : GetHeight(node.Left) - GetHeight(node.Right);

        public int Size(Node root)
        {
            if (root == null)
                return 0;
            return 1 + Size(root.Left) + Size(root.Right);
        }

        public void AdjustHeight(Node node)
        {
            if (node != null)
                node.Height = Math.Max(GetHeight(node.Left), GetHeight(node.Right)) + 1;
        }

        public void RotateRight(Node x)
        {
            if (x == null)
                return;

            Node p = x.Parent;
            Node y = x.Left;
            Node b = y != null ? y.Right : null;
            if (y != null)
                y.Parent = p;

            if (p != null)
            {
                if (p.Left == x)
                    p.Left = y;
                else
                    p.Right = y;
            }

            x.Parent = y;
            if (y != null)
                y.Right = x;
            if (b != null)
                b.Parent = x;
            x.Left = b;

        }

        public void RotateLeft(Node x)
        {
            if (x == null)
                return;

            Node p = x.Parent;
            Node y = x.Right;
            Node b = y != null ? y.Left : null;
            if (y != null)
                y.Parent = p;

            if (p != null)
            {
                if (p.Left == x)
                    p.Left = y;
                else
                    p.Right = y;
            }

            x.Parent = y;
            if (y != null)
                y.Left = x;
            if (b != null)
                b.Parent = x;
            x.Right = b;
        }

        public void RebalanceRight(Node n)
        {
            if(n==null)
                return;
            Node M = n.Left;
            if (M != null && GetHeight(M.Right) > GetHeight(M.Left))
            {
                RotateLeft(M);
                AdjustHeight(M);
            }
            RotateRight(n);
            AdjustHeight(n);

        }

        public void RebalanceLeft(Node n)
        {
            if(n==null)
                return;
            Node M = n.Right;
            if (M != null && GetHeight(M.Left) > GetHeight(M.Right))
            {
                RotateRight(M);
                AdjustHeight(M);
            }
            RotateLeft(n);
            AdjustHeight(n);
        }

        public void Rebalance(Node n)
        {
           if(n==null)return;
            Node P = n.Parent;
            if (GetHeight(n.Left) > GetHeight(n.Right) + 1)
            {
                RebalanceRight(n);
            }
            if (GetHeight(n.Right) > GetHeight(n.Left) + 1)
            {
                RebalanceLeft(n);
            }
            AdjustHeight(n);
            if(P!= null)
            {
                Rebalance(P);
            }
        }

        public Node Find(Box k, Node r)
        {
            if (r.Box.Capacity == k.Capacity)
                return r;
            else if (r.Box.Capacity > k.Capacity)
            {
                if (r.Left != null)
                    return Find(k, r.Left);
                else
                {
                    return r;
                }
            }
            else if(r.Box.Capacity < k.Capacity)
            {
                if (r.Right != null)
                    return Find(k, r.Right);
                else
                    return r;
            }
            return r;
        }

        public Node Insert(Box k,Node root)
        {
            Node P = Find(k, root);
             Node ins= new Node(k);
            if (P!=null)
            {
                if (P.Box.Capacity > k.Capacity)
                {
                    P.Left = ins;
                    
                }
                else
                {
                    P.Right = ins;
                }
                ins.Parent = P;
                
            }
            return ins;
        }

        public void AVLInsert(Box k, Node root)
        {
           Node n=Insert(k, root);
            Rebalance(n);
        }

        public Node Next(Node n)
        {
            if (n.Right != null)
            {
                n = n.Right;
                while (n.Left != null)
                    n = n.Left;
                return n;
            }
            while (n.Parent != null&&n.Box.Capacity>n.Parent.Box.Capacity)
                n = n.Parent;
            return n.Parent;
        }
        private void promote(Node x,Node y)
        {
            y.Parent = x.Parent;
            if(x.Parent.Left==x)
               x.Parent.Left = y;
            else
                x.Parent.Right = y;
            x.Right = null;
            x.Left = null;
            x.Parent = null;
        }

        public Node Prev(Node n)
        {
            if (n.Left != null)
            {
                n = n.Left;
                while (n.Right != null) n = n.Right;
                return n;
            }
            while (n.Parent != null && n.Parent.Left == n) n = n.Parent;
            return n.Parent;
        }//לבדוק אם יוצא להשתמש

        public void Swap(Node a, Node b)
        {
            if (a == null || b == null || a == b) return;
            Box temp = a.Box;
            a.Box = b.Box;
            b.Box = temp;
        }//כנל

        public Node Delete(Node n)
        {
            if (n == null) return null; // בדיקה אם הצומת הוא null

            if (n.Left == null || n.Right == null) // only one child or no child
            {
                Node temp = n.Left != null ? n.Left : n.Right;

                if (temp == null) // No children (leaf)
                {
                    if (n.Parent != null) // אם יש הורה
                    {
                        if (n.Parent.Left == n)
                            n.Parent.Left = null;
                        else if (n.Parent.Right == n)
                            n.Parent.Right = null;
                    }
                    Node replacement = n.Parent; // מחזירים את ההורה כמחליף (אם יש)
                    n.Parent = null; // ניתוק הצומת
                    return replacement;
                }
                else // יש ילד אחד
                {
                    temp.Parent = n.Parent; // קישור ההורה של temp להורה של n

                    if (n.Parent != null) // אם יש הורה
                    {
                        if (n.Parent.Left == n)
                            n.Parent.Left = temp;
                        else if (n.Parent.Right == n)
                            n.Parent.Right = temp;
                    }
                    // ניתוק הצומת
                    n.Parent = null;
                    n.Left = null;
                    n.Right = null;
                    return temp;
                }
            }
            else // two children
            {
                Node next = Next(n); // היורש במקום

                if (next != null)
                {
                    if (next.Right != null) // next from sub right tree
                    {
                        n.Box = next.Box; // העתקת ערך
                        promote(next, next.Right); // קידום
                    }
                    else if (next.Left != null) // right ancestor-next (מקרה נדיר יותר, אך הכנסנו לפי הקוד שלך)
                    {
                        if (n.Parent != null)
                        {
                            if (n.Parent.Left == n)
                                n.Parent.Left = n.Left;
                            else if (n.Parent.Right == n)
                                n.Parent.Right = n.Left;
                        }

                        n.Left.Parent = n.Parent;

                        Node replacement = n.Left;

                        n.Parent = null;
                        n.Left = null;
                        n.Right = null;
                        return replacement;
                    }
                    else // next הוא עלה
                    {
                        n.Box = next.Box; // מחליפים ערך
                        if (next.Parent != null)
                        {
                            if (next.Parent.Left == next)
                                next.Parent.Left = null;
                            else if (next.Parent.Right == next)
                                next.Parent.Right = null;
                        }
                        next.Parent = null;
                        return n;
                    }
                }
            }

            return null; 
        }

        public Node DeleteAVL(Node n)
        {
            Node replacment = Delete(n);
            if(replacment != null)
            {
                if(replacment.Left!=null)
                    Rebalance(replacment.Left);
                else
                {
                    Rebalance(replacment);
                }
            }
            return n;
        }
    }
}