using EasyPack.Core.Models;
using EasyPack.Core.Repositories;
using EasyPack.Core.Service;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Service.Service
{
    public class UserService:IUserService
    {
        private readonly IUserRepository _userRepository;

        public UserService(IUserRepository userRepository)
        {
            this._userRepository = userRepository;
        }

        public List<User> GetUserList()
        {
            return _userRepository.GetUserList();
        }

        public User GetUserById(int id)
        {
            return _userRepository.GetUserById(id);
        }
        public User GetUserByEmail(string email)
        {
            return _userRepository.GetUserByEmail(email);
        }

        public void DeleteUser(int id)
        {
            _userRepository.DeleteUser(id);
        }

        public User UpdateUser(User user, int id)
        {
            return _userRepository.UpdateUser(user, id);
        }

        public User AddUser(User user)
        {
            return _userRepository.AddUser(user);
        }
    }
}
